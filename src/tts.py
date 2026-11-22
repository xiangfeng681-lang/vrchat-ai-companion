"""嗓子:文字 → 波形 → 播放到虚拟声卡。

**三个后端**,由 `tts.backend` 决定:

- `edge` —— 微软在线合成。免费、不占显存,但每次都有 **~1.85s 的固定握手开销**
  (跟句子长短无关),而且**不能克隆音色**,只能选微软那几个。
- `gpt_sovits` —— 本地克隆。吃 2~3GB 显存,开销**与句长成正比**(RTF≈0.48),
  所以短句更快、长句更慢。换来的是**任意音色克隆**和**不怕断网**。
- `cosyvoice` —— 阿里云百炼**云端克隆**。零显存、能克隆任意音色(先在云端注册成
  一个 `voice_id`),⚠️ 代价是**断网就哑**。迁到 6G 显存的机器就靠它。

链路:文字 → 音频字节 → 解码(soundfile)→ 重采样到设备原生率 → 分块写入输出设备

顺带做了口型:播放时按 50ms 一块算音量,回调上层去驱动 Avatar 的嘴。
不需要音素时间戳,音量包络对「嘴巴开合」这种需求已经足够自然。

依赖说明:edge-tts 出 mp3、百炼 cosyvoice 默认也出 mp3,soundfile 自带的
libsndfile 1.2+ 能直接解,所以**不需要装 ffmpeg**。如果你的 soundfile 版本
太老解不了 mp3,报错里会提到 format,那时再考虑 `pip install -U soundfile`。

⚠️ **三个后端的失败语义**:一律 raise,由 `Pipeline._speak_loop` 兜住
(记 `TTS错误`、嘴巴归零、跳过这一句)。**刻意不做兜底** —— 曾经想退到 edge,
但实测 edge-tts 在中文网络环境下持续 403 不可用,兜底等于白等一次再失败;
而抛异常是安全的。**

🔑 **为什么抛异常在这里是安全的**:`on_ready`(发字幕、做动作)挂在
`synthesize()` **返回之后**。抛异常就到不了那一步,所以**不会出现
"有字幕没声音"的哑剧** —— 那句直接从对话里消失,而不是演出来。
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.request

import edge_tts
import numpy as np
import soundfile as sf
import sounddevice as sd

from .audio import devices
from .audio.resample import resample
from .config import TtsConfig

# 语音的 RMS 通常在 0.05~0.15,乘个系数才适合当「嘴巴张开度」
_LEVEL_GAIN = 6.0
_CHUNK_SECONDS = 0.05

# 云端 TTS 熔断后冷却多久。够长到"别一句一卡",够短到"网络回来了能自愈"。
_CLOUD_BREAKER_COOLDOWN_S = 60.0


class TTS:
    def __init__(self, cfg: TtsConfig, output_device=None):
        self.cfg = cfg
        self.device_index = devices.resolve(output_device, "output")
        self._channels: int | None = None
        self._rate: int | None = None
        # 云端(cosyvoice)的熔断状态,见 _cloud_usable
        self._cloud_fail = 0
        self._cloud_block_until = 0.0
        self.last_note = ""          # 最近一次降级/不可用的原因,给上层记日志

    @property
    def name(self) -> str:
        return devices.describe(self.device_index)

    # ---------------- 合成 ----------------

    def _to_mono(self, data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)

    @staticmethod
    def _detect_lang(text: str) -> str:
        """粗略判断这段文字该用哪个语言的 G2P 前端。

        不用 langdetect 那种库 —— GPT-SoVITS 只有 zh/ja/en/ko/yue 几个选项,
        而**中日文都用汉字**,真正的判据是**假名**:
          - 出现平假名/片假名 → 一定是日语
          - 没有假名、汉字比拉丁字母多 → 中文
          - 拉丁字母占多数 → 英语
        混杂时按数量取多数,够用了。
        """
        has_kana = any("぀" <= c <= "ヿ" for c in text)
        if has_kana:
            return "ja"
        cjk = sum(1 for c in text if "一" <= c <= "鿿")
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        return "zh" if cjk >= latin else "en"

    def _normalize(self, text: str, lang: str = "zh") -> str:
        """合成前的文本替换。

        中英混排会让 G2P 在语言边界插一个停顿 —— 实测「小mo」比「小茉」
        长 0.68s,听感上是"小……mo"。**只替换送去合成的文本**,字幕和
        对话记录里仍然是原样,用户看不出区别。
        """
        # 只对**中文**做替换。「小mo→小茉」是中文的同音字修正(躲开中英混排的
        # 停顿),日语里把"小mo"换成"小茉"反而会让发音完全变样。
        if lang != "zh":
            return text
        out = text
        for src, dst in (self.cfg.name_fix or {}).items():
            out = out.replace(src, dst)
        return out

    async def _synth_edge(self, text: str, path: str) -> None:
        comm = edge_tts.Communicate(
            text, self.cfg.voice, rate=self.cfg.rate, volume=self.cfg.volume
        )
        await comm.save(path)

    def _synth_edge_bytes(self, text: str) -> tuple[np.ndarray, int]:
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="vrchat_tts_")
        os.close(fd)
        try:
            asyncio.run(self._synth_edge(text, path))
            if os.path.getsize(path) == 0:
                raise RuntimeError(f"Edge-TTS 返回了空音频(文本={text!r}),检查网络或音色名")
            data, sr = sf.read(path, dtype="float32", always_2d=False)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return self._to_mono(data, sr)

    def _synth_gpt_sovits(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        """打本地 GPT-SoVITS 的 /tts 接口,拿回 wav 字节。

        ⚠️ `prompt_text` 必须是参考音频里**真实说的那句话**。写错了音色会
        明显跑偏 —— 它不是"随便给段文字",是让模型对齐参考音频用的。
        """
        if not self.cfg.ref_audio:
            raise RuntimeError("tts.ref_audio 没配 —— GPT-SoVITS 需要一段参考音频才能克隆音色")
        payload = {
            "text": text,
            # 语言**自动判断** —— 写死 "zh" 的话,它回复日语/英语时也会被
            # 当中文念,发音全错。GPT-SoVITS 的 v2ProPlus 同一个模型支持
            # 中/日/英/韩/粤,切语言只是换这个参数,**不加载新模型、不加显存**。
            "text_lang": lang,
            "ref_audio_path": self.cfg.ref_audio,
            "prompt_text": self.cfg.prompt_text,
            "prompt_lang": self.cfg.prompt_lang,
            "text_split_method": "cut5",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": False,
            "speed_factor": self.cfg.speed_factor,
            "temperature": self.cfg.temperature,
            "top_k": self.cfg.top_k,
            "top_p": self.cfg.top_p,
            "seed": -1,
        }
        req = urllib.request.Request(
            self.cfg.api_url.rstrip("/") + "/tts",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"连不上 GPT-SoVITS 服务({self.cfg.api_url}) —— 先把 api_v2.py 起起来。{exc}"
            ) from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"GPT-SoVITS 报错 {exc.code}: {detail}") from exc

        if not raw:
            raise RuntimeError(f"GPT-SoVITS 返回空音频(文本={text!r})")
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return self._to_mono(data, sr)

    # ---------------- 云端(阿里云百炼 cosyvoice) ----------------
    #
    # 和本地 GPT-SoVITS 的**根本区别**:sovits 是 zero-shot,每次合成现场喂一遍
    # 参考音频;云端是**先注册成一个 voice_id**,之后只传这个 ID。
    # 注册是一次性的,见 tools/enroll_dashscope_voice.py。

    def _cloud_usable(self) -> bool:
        """云端现在该不该试?—— 熔断器。

        断网时云端每次都要白等 cloud_timeout_s。不熔断的话,LLM 一口气吐 5 句,
        小mo 就要先卡 150 秒才开口。所以连着失败 cloud_breaker_n 次就"熔断"
        一段时间,期间直接走 edge;到点后放一次请求试探(半开)。
        """
        c = self.cfg
        if not c.cloud_voice_id:
            self.last_note = "cloud_voice_id 没配"
            return False
        if not os.environ.get(c.cloud_api_key_env):
            self.last_note = f"没设 {c.cloud_api_key_env}"
            return False
        if time.monotonic() < self._cloud_block_until:
            self.last_note = "云端熔断中"
            return False
        return True

    def _cloud_ok(self) -> None:
        self._cloud_fail = 0
        self._cloud_block_until = 0.0
        self.last_note = ""

    def _cloud_failed(self, exc: Exception) -> None:
        self._cloud_fail += 1
        self.last_note = f"云端 TTS 失败({self._cloud_fail}): {type(exc).__name__}"
        if self._cloud_fail >= self.cfg.cloud_breaker_n:
            self._cloud_block_until = time.monotonic() + _CLOUD_BREAKER_COOLDOWN_S
            self.last_note += f" → 熔断 {_CLOUD_BREAKER_COOLDOWN_S:.0f}s"

    def _synth_cosyvoice(self, text: str) -> tuple[np.ndarray, int]:
        """打阿里云百炼的 cosyvoice,音色是注册好的 voice_id。

        走官方 `dashscope` 包 —— **不是** OpenAI 兼容接口。
        实测:兼容接口的 `/audio/speech` 是 **404**(端点根本不存在),
        而原生 HTTP 端点上 `cosyvoice` 系列也不合成。官方 SDK 的
        `tts_v2.SpeechSynthesizer` 是唯一走通的通道。

        🔴 三个必须逐字对齐的东西,错一个就报错:
          1. `voice` 填**注册拿到的 voice_id**(cosyvoice-v2-xxx-<hex>),
             不是 Cherry/longxiaochun 这些官方音色名。
          2. `model` 必须和注册时的 `target_model` **完全一致**。
          3. 音色**不能跨模型用** —— 用 cosyvoice-v2 注册的音色,
             拿去 cosyvoice-v3 合成是不行的。

        ⚠️ **不在这里兜底** —— 异常一律上抛,由 synthesize() 决定要不要退回 edge。
        """
        c = self.cfg
        if not c.cloud_voice_id:
            raise RuntimeError(
                "tts.cloud_voice_id 没配 —— 先跑 tools/enroll_dashscope_voice.py "
                "注册音色,再把打印出来的 voice_id 填进 config.yaml"
            )
        key = os.environ.get(c.cloud_api_key_env)
        if not key:
            raise RuntimeError(
                f"没设环境变量 {c.cloud_api_key_env} —— "
                f"`setx {c.cloud_api_key_env} sk-xxx` 后重开终端"
            )

        import dashscope                       # 懒 import:不用云端就不付这点开销
        from dashscope.audio.tts_v2 import SpeechSynthesizer

        dashscope.api_key = key
        try:
            syn = SpeechSynthesizer(model=c.cloud_model, voice=c.cloud_voice_id)
            # call() 是**阻塞**的,返回完整音频字节 —— 正好是 synthesize() 要的形状,
            # 所以 pipeline / speak / play / on_ready 全都不用改。
            raw = syn.call(text, timeout_millis=int(c.cloud_timeout_s * 1000))
        except Exception as exc:
            raise RuntimeError(f"百炼 cosyvoice 失败({type(exc).__name__}): {exc}") from exc

        if not raw:
            raise RuntimeError(f"百炼 cosyvoice 返回空音频(文本={text!r})")
        # 默认格式是 mp3/22050Hz 单声道,libsndfile 1.2+ 直接解(和 edge 那条路同一套)
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return self._to_mono(data, sr)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """返回 (单声道 float32 波形, 采样率)。

        ⚠️ **`cosyvoice` 失败时直接抛异常,不兜底**(见模块 docstring)。
        抛异常是安全的:pipeline 会记 `TTS错误`、嘴巴归零、跳过这一句,
        而且 **on_ready 根本不会触发** —— 所以不会出现"有字幕没声音"的哑剧。
        """
        lang = self._detect_lang(text)
        text = self._normalize(text, lang)

        if self.cfg.backend == "cosyvoice":
            if not self._cloud_usable():
                # 没配 voice_id / 没设 key / 熔断中 —— 立刻失败,别等超时
                raise RuntimeError(f"云端 TTS 不可用:{self.last_note}")
            try:
                out = self._synth_cosyvoice(text)
                self._cloud_ok()
                return out
            except Exception as exc:
                self._cloud_failed(exc)
                raise

        if self.cfg.backend == "gpt_sovits":
            return self._synth_gpt_sovits(text, lang)
        return self._synth_edge_bytes(text)

    # ---------------- 播放 ----------------

    def _output_channels(self) -> int:
        if self._channels is None:
            info = sd.query_devices(self.device_index)
            self._channels = min(2, max(1, int(info["max_output_channels"])))
        return self._channels

    def _output_rate(self) -> int:
        if self._rate is None:
            self._rate = int(sd.query_devices(self.device_index)["default_samplerate"])
        return self._rate

    def play(
        self,
        data: np.ndarray,
        sample_rate: int,
        on_level=None,
        should_stop=None,
    ) -> bool:
        """分块播放,每块回调一次音量(0~1),用于驱动口型。

        ⚠️ 必须先把采样率对齐到设备原生率:Edge-TTS 出的是 24000Hz,
        而 WASAPI 共享模式只收设备原生格式(通常 48000Hz),不转会直接报
        `Invalid sample rate [PaErrorCode -9997]`。

        `should_stop` 是"停止词"用的打断钩子 —— 每个块(50ms)问一次,
        返回 True 就立刻掐断。50ms 的粒度人耳感觉不到延迟。
        返回 True 表示被打断了,False 表示正常放完。
        """
        target = self._output_rate()
        if sample_rate != target:
            data = resample(data, sample_rate, target)
            sample_rate = target

        channels = self._output_channels()
        step = max(1, int(sample_rate * _CHUNK_SECONDS))
        interrupted = False

        with sd.OutputStream(
            device=self.device_index,
            channels=channels,
            samplerate=sample_rate,
            dtype="float32",
        ) as stream:
            for start in range(0, len(data), step):
                if should_stop is not None and should_stop():
                    interrupted = True
                    break
                block = data[start:start + step]
                if on_level is not None:
                    level = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
                    on_level(min(1.0, level * _LEVEL_GAIN))
                if channels > 1:
                    block = np.repeat(block.reshape(-1, 1), channels, axis=1)
                stream.write(block)

        if on_level is not None:
            on_level(0.0)
        return interrupted

    @staticmethod
    def _speakable(text: str) -> bool:
        """这段文字到底有没有"能念出来的东西"。

        LLM 偶尔会甩出一个只有标点的"句子" —— 「…」「!!!」「，。」。
        GPT-SoVITS 会直接返回 **HTTP 400 请输入有效文本**,那一句白等一次请求。
        Edge-TTS 更糟:它会老老实实去合成一段静音,听起来像卡了一下。
        所以这类片段在**送去合成之前**就丢掉。
        """
        return any(ch.isalnum() or "一" <= ch <= "鿿" for ch in text)

    def speak(self, text: str, on_level=None, on_ready=None, should_stop=None) -> bool:
        """合成 → on_ready() → 播放。返回 True 表示被打断了。

        ⚠️ `on_ready` 是**字幕该挂的地方** —— 它在"合成完成、即将出声"的
        那一刻触发,字幕和声音才会同时出现。挂早了(比如收到句子时就发)字幕会
        比声音早整整一个合成的时长(本地克隆约 2s),观感很割裂。
        """
        if not text.strip() or not self._speakable(text):
            return False
        data, sr = self.synthesize(text)
        if on_ready is not None:
            on_ready()
        return self.play(data, sr, on_level, should_stop=should_stop)
