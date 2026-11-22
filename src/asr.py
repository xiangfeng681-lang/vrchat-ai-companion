"""耳朵的翻译层:音频 → 文字(faster-whisper)。

faster-whisper 是重依赖(拉 ctranslate2 / onnxruntime),所以**延迟导入** ——
前面几个阶段(osc / chat / tts)不需要装它也能跑。

⚠️ Whisper 只吃 **16kHz**。回环采集拿到的是 48000Hz(Windows 共享模式的硬性
   要求),中间必须重采样,否则会把正常人声听成慢速低音。这里用 soxr 做,
   没有 soxr 就退化成线性插值(有混叠,但能凑合)。

首次调用会从 HuggingFace 下载模型:
    tiny≈75MB  base≈145MB  small≈480MB  medium≈1.5GB
下载慢的话设环境变量 HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import difflib
import time

import numpy as np

from .audio.resample import resample
from .config import AsrConfig

TARGET_RATE = 16000

# 匹配用的**繁→简**映射表。
#
# ⚠️ 为什么需要它:`asr.language` 改成 `auto` 之后,中文提示词
#    (「请用简体中文转写」)就不再送进 Whisper 了 —— 而那个提示词存在的
#    全部意义就是**把中文钉在简体上**。代价就是繁体回潮(实测 7 句里 4 句繁体)。
#
#    LLM 读繁体毫无问题,真正会坏的是**字符串匹配**:
#    停止词配的是「闭嘴」,ASR 吐的是「閉嘴」—— 一个字都对不上,
#    说"闭嘴"它根本不会停。唤醒词(`小莫`/`小茉`/`小墨`)繁简同形,不受影响。
#
# 只收**本项目和配置里真实出现过**的字,不做全量转换 ——
# 真要通用转换该上 opencc,那里有一万多个字的对照表。这里是够用就好。
SIMPLIFY = str.maketrans({
    # 停止词:wake.stop_words
    "閉": "闭", "別": "别", "說": "说", "靜": "静",
    # 媒体残留黑名单:_MEDIA_ARTIFACTS / _MEDIA_PHRASES
    "謝": "谢", "觀": "观", "見": "见", "組": "组", "請": "请",
    "點": "点", "讚": "赞", "訂": "订", "閱": "阅", "轉": "转",
    "發": "发", "賞": "赏", "鏡": "镜", "與": "与",
})


def to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """把任意采样率的单声道 float32 重采样到 16k。"""
    return resample(audio, src_rate, TARGET_RATE)


# 云端 ASR 的提示词。两个要点:
#   ① **只要转写结果** —— 不交代清楚,它会加「这段音频的内容是…」这种包装,
#      那会直接被当成用户说的话。
#   ② 🔴 **听不清时要一个固定标记** —— 全模态模型在没听清时会**编客套话**
#      (「抱歉,我这边没有听到任何声音」),而那句话会被当成"用户说了这句",
#      比转写不准严重得多。所以约定:没人说话/听不清 → 只输出「空」。
_CLOUD_ASR_PROMPT = (
    "把这段音频一字不差地转写出来。只输出转写结果本身 —— "
    "不要解释、不要加引号、不要描述音频、不要加任何前后缀。"
    "如果音频里没有人说话、或者听不清,就只输出一个字:空"
)
_CLOUD_ASR_EMPTY = {"空", "", "（空）", "(空)", "..."}


def _cloud_prompt(hint: str) -> str:
    """转写提示词 + 可选的**偏置提示**。

    🔑 **这段 hint 是拿回"热词偏置"的唯一办法。**

    本地 Whisper 那条路有 `hotwords` 参数(记忆里记着:"🔑 **热词偏置** ——
    解决'叫了没反应'的正解",当时把被吞成「那」的「小莫」救了回来)。
    但**云端走的是 chat 接口,没有 hotwords** —— 换成 `backend: cloud` 之后
    那层偏置就丢了,表现为「叫了没反应」又回来,而且**别人的嗓音更明显**
    (自己的声音它听惯了)。

    云端模型能接受的只有提示词,所以只能这样"预告"可能会出现的词。
    ⚠️ **hint 要短** —— 这个模型偏"想",提示词越长它越容易跑偏
    (眼睛那个模型栽过:提示词太长,token 全烧在思考上、正文全空)。
    """
    if not hint:
        return _CLOUD_ASR_PROMPT
    return _CLOUD_ASR_PROMPT + "\n" + hint


class Transcriber:
    def __init__(self, cfg: AsrConfig):
        self.cfg = cfg
        self._model = None
        self.device = cfg.device
        self.compute_type = cfg.compute_type
        # 最近一次转写 Whisper 判定出来的语言和它的置信度。
        # 只在 language=auto 时有意义 —— 写死语言时 Whisper 不做判断。
        # 调用方(pipeline)拿它打日志:自动识别**必须看得见**,
        # 否则中文被认成日文、日文被认成中文,谁都发现不了。
        self.last_lang = ""
        self.last_lang_prob = 0.0
        # 最近几次转写的原文和时间 —— 给「同一句反复出现」那道闸记账用。
        self._recent: list[tuple[float, str]] = []
        # 上一次转写被哪道闸丢掉的。空串 = 没丢。
        # 用处:日志里 12/18 都是"空结果"时,**得知道是为什么空的** ——
        # 是静音、是幻觉、还是闸门太狠误杀了真话。不区分就只能瞎猜。
        self.last_drop = ""
        # 被"无人声"闸丢掉时,那一段里最高的 no_speech_prob。
        # 判断"阈值太严"还是"确实没人在说"全靠这个数(见 transcribe)。
        self.last_nospeech = 0.0

    def _prepare_hf_env(self) -> None:
        """把 HuggingFace 下载指向镜像,并关掉 Xet 通道。

        ⚠️ 两个坑,踩过:

        1. **镜像要主动设,不能等它自己出现。** 这里原来只在「HF_ENDPOINT 已存在」
           时才关 Xet —— 结果在这台**连不上 huggingface.co** 的机器上,首次下模型
           必然 `ConnectTimeout`,除非你记得手动 export。现在默认走镜像,
           想用官方源就把 `asr.hf_endpoint` 设成空串。

        2. **设了镜像就必须关 Xet。** huggingface_hub 新版默认走 Xet,下载源是
           cas-server.xethub.hf.co —— 镜像站**不代理**那个域名,不关就下到一半
           报 401 Unauthorized。
        """
        import os

        endpoint = (self.cfg.hf_endpoint or "").strip()
        if endpoint and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = endpoint
        if os.environ.get("HF_ENDPOINT") and "HF_HUB_DISABLE_XET" not in os.environ:
            os.environ["HF_HUB_DISABLE_XET"] = "1"

    @staticmethod
    def _enable_cuda_dlls() -> None:
        """把 pip 装的 nvidia-cublas / nvidia-cudnn 的 DLL 目录挂进搜索路径。

        ctranslate2 不会自己找它们。不挂的话会报
        `cublas64_12.dll is not found or cannot be loaded`。
        (装了 CUDA Toolkit 并用 system-wide DLL 的话这段是空操作。)

        ⚠️ 两个必须踩对的点,否则**静默退回 CPU**,很难查:

        1. 必须在 `import ctranslate2` / `faster_whisper` **之前**调用。
           ctranslate2 在导入时就完成 CUDA 后端初始化,之后再挂 DLL 已经晚了。
           `_load()` 里的调用位置是刻意的,别往下挪。

        2. 必须用 importlib.import_module,不能用 getattr(nvidia, pkg):
           nvidia 是 PEP 420 命名空间包(__file__ 为 None),`import nvidia`
           不会顺带导入子包,getattr 永远拿到 None。

        add_dll_directory 和 PATH 都挂:ctranslate2 内部走的是裸 LoadLibrary,
        它只认 PATH,不认 add_dll_directory 注册的目录。
        """
        import importlib
        import os
        import sys

        if sys.platform != "win32":
            return
        dirs: list[str] = []
        for pkg in ("cublas", "cudnn", "cuda_runtime"):
            try:
                mod = importlib.import_module(f"nvidia.{pkg}")
            except ImportError:
                continue
            for base in getattr(mod, "__path__", []):
                bin_dir = os.path.join(base, "bin")
                if os.path.isdir(bin_dir):
                    dirs.append(bin_dir)
                    try:
                        os.add_dll_directory(bin_dir)
                    except OSError:
                        pass
        if dirs:
            os.environ["PATH"] = ";".join(dirs) + ";" + os.environ.get("PATH", "")

    @staticmethod
    def _probe(model) -> None:
        """跑一次最小推理,确认这个 device 真的能用。

        ⚠️ 必须做:transcribe() 返回的是**惰性生成器**,不消费就永远不推理;
        而 CUDA 的 DLL 缺失恰恰只在推理时才报错,装载时完全看不出来。
        """
        segments, _ = model.transcribe(np.zeros(TARGET_RATE, dtype=np.float32), beam_size=1)
        list(segments)

    def _load(self):
        if self._model is not None:
            return self._model

        self._prepare_hf_env()    # 必须在 import faster_whisper 之前设好
        self._enable_cuda_dlls()  # 必须在 import ctranslate2 之前,别往下挪

        from faster_whisper import WhisperModel  # 延迟导入

        device, compute = self.device, self.compute_type
        if device == "auto":
            device = "cpu"
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                pass
        if device == "cpu" and compute == "float16":
            compute = "int8"   # CPU 不支持 fp16
        if device == "cuda" and compute == "int8":
            # 配置里的 int8 是给 CPU 的默认值。device=auto 自动选了显卡时,
            # 沿用 int8 会在 GPU 上跑最慢的那档 kernel;换 float16 才是显卡该有的速度。
            compute = "float16"

        model = WhisperModel(self.cfg.model, device=device, compute_type=compute)

        try:
            self._probe(model)
        except Exception as exc:
            if device != "cuda":
                raise
            print(f"[ASR] 检测到显卡但 CUDA 跑不起来({exc})")
            print("[ASR] 退回 CPU。想用显卡的话:")
            print("      pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
            device, compute = "cpu", "int8"
            model = WhisperModel(self.cfg.model, device=device, compute_type=compute)
            self._probe(model)

        self.device, self.compute_type = device, compute
        self._model = model
        return model

    def warmup(self) -> None:
        """提前加载模型,避免第一次说话时卡好几秒。"""
        self._load()

    # 低于这个 RMS 直接当静音。Whisper 喂静音**不会**返回空,它会开始编。
    SILENCE_RMS = 0.0015
    # segment 被判为「不是语音」的概率超过这个值就丢掉。
    # 实测纯噪声/静音会给 0.83~0.86(Whisper 自己知道没人说话),真人语音远低于此。
    # ⚠️ 这道闸的阈值现在在 config.yaml 的 `asr.no_speech_max`(默认 0.8)。
    # 以前是这里的常量、写死 0.6 —— 太保守,实测把真人说话误杀过。
    # 详见 AsrConfig.no_speech_max 的注释。
    # 与 initial_prompt 的相似度超过这个值就当成提示词幻觉(见下)。
    # 阈值是量出来的:实测幻觉 0.60~0.64,真实语音最高 0.11,中间空一大片。
    PROMPT_ECHO_RATIO = 0.45

    # ---- 第五道闸:同一句反复出现 ----
    # 这是**机制**,不是穷举。Whisper 的中文幻觉写法多到黑名单永远补不完
    # (「请不吝点赞 订阅 转发 打赏支持明镜与点点栏目」只是被抓到的那一种),
    # 但它们有个共同点:**在音乐/静音上一遍遍吐同一句**。
    # 真人对话不会在 2 分钟里把同一句 8 字以上的话原样重复 3 遍。
    REPEAT_MIN_CHARS = 8        # 短于此不参与判断(「对,对,对。」是真人会说的)
    REPEAT_WINDOW_S = 120.0     # 统计窗口
    REPEAT_MAX = 3              # 第 3 次开始丢掉(前两次放行,见 _looks_like_repeat)

    def _looks_like_prompt_echo(self, text: str) -> bool:
        """判断这句是不是 Whisper 在续写 initial_prompt。

        喂静音/音乐时 Whisper 会把 initial_prompt 当成"上文"往下编。实测两种形态:

        - 原样照抄:「请用简体中文转写。」
        - **改写后再吐**:「这句话是普通话的词,请用简体中文转写。什么,你是什么大母亲?」

        第二种最坑 —— 它是提示词的**改写**,所以「子串包含」判据抓不住,
        必须用模糊相似度。好在量下来区分度极大(幻觉 0.60+ / 真语音 <=0.11),
        卡 0.45 既稳又不误杀。
        """
        prompt = self.cfg.initial_prompt or ""
        if not prompt or not text:
            return False
        strip = lambda s: "".join(c for c in s if c.isalnum())  # noqa: E731
        t, p = strip(text), strip(prompt)
        if len(t) < 4:
            return False
        if t in p or p in t:
            return True
        return difflib.SequenceMatcher(None, t, p).ratio() > self.PROMPT_ECHO_RATIO

    # Whisper 在音乐/杂音上最常编出来的媒体残留。这类跟用户的语音毫无关系,
    # 而且 no_speech_prob 往往不高(音乐让它以为"有内容"),只能靠黑名单。
    # 只收明确的媒体署名类,别往里加「订阅」「点赞」这种真人也会说的词。
    _MEDIA_ARTIFACTS = ("字幕by", "字幕由", "字幕组", "谢谢观看", "谢谢大家观看",
                        "感谢观看", "下期见", "下次见", "MING PAO")
    # 只有整句都很短时才算残留 —— 长句里出现这些词可能是用户真在聊这话题。
    _ARTIFACT_MAX_LEN = 14

    # **整句式**片尾 —— 这类是 Whisper 在音乐/静音上编出来的**成句**残留,
    # 真人不会在 VRChat 里整句念出来。**不受 _ARTIFACT_MAX_LEN 限制**。
    #
    # 为什么单开一张表:实测「请不吝点赞 订阅 转发 打赏支持明镜与点点栏目」
    # 去掉空格有 **20 字**,卡在 14 字闸外面 —— 4 分钟里漏了 7 次。
    # 长度闸的本意是"短句才可能是残留",但媒体片尾**天生就长**,
    # 用长度当判据在这类上是反的。
    #
    # ⚠️ 还是别往里加「订阅」「点赞」这种**单词** —— 真人真会聊这个。
    #    要加就加**整串**,越长越安全。
    _MEDIA_PHRASES = ("请不吝点赞", "订阅转发打赏", "明镜与点点")

    def _looks_like_media_artifact(self, text: str) -> bool:
        # 繁简归一是**必须**的:黑名单配的是简体,而 auto 模式下 ASR 会吐繁体
        # (实测漏过一条「多謝您的觀看,下次見!」)。不归一就得把每一条都写成两份。
        t = "".join(c for c in text.translate(SIMPLIFY) if c.isalnum())
        if not t:
            return False
        low = t.lower()
        # 整句式片尾:不管多长,命中就丢
        if any(p.replace(" ", "").lower() in low for p in self._MEDIA_PHRASES):
            return True
        # 短签名:仍然只对短句生效 —— 长句里出现这些词可能是在聊这话题
        if len(t) > self._ARTIFACT_MAX_LEN:
            return False
        return any(a.replace(" ", "").lower() in low for a in self._MEDIA_ARTIFACTS)

    def looks_like_speech(self, text: str) -> bool:
        """这段转写值不值得当成「有人在说话」。

        给**旁听通道**用。旁听走的是 `quick=True`,而 quick 通道刻意不过幻觉闸
        (见 transcribe 里 `if quick: return text`)—— 平时无所谓,但一旦拿它做
        「有人说话就让路」的判断,不把关就会**一直让路**:音乐/静音上编出来的
        片尾会被当成"别人在说话"。
        """
        t = "".join(c for c in text if c.isalnum())
        if len(t) < 2:
            return False
        return not self._looks_like_media_artifact(text)

    def _looks_like_repeat(self, text: str) -> bool:
        """同一句在短时间内反复出现 → 判为幻觉。**同时负责记账。**

        ⚠️ 前 `REPEAT_MAX - 1` 次**放行**。第一遍没办法知道它是不是幻觉 ——
        只能等它重复。宁可漏放前两次(最多多两句噪音),也不要冤杀第一句真话。
        """
        t = "".join(c for c in text if c.isalnum())
        if len(t) < self.REPEAT_MIN_CHARS:
            return False
        now = time.monotonic()
        # 先清理过期记录,顺便防止列表无限长
        self._recent = [(ts, s) for ts, s in self._recent
                        if now - ts <= self.REPEAT_WINDOW_S]
        if sum(1 for _, s in self._recent if s == t) >= self.REPEAT_MAX - 1:
            return True
        self._recent.append((now, t))
        return False

    def _resolve_language(self) -> str | None:
        """把配置里的 `language` 翻成 faster-whisper 要的参数。

        `"auto"` / 空 → `None`,交给 Whisper 自己判。
        """
        lang = (self.cfg.language or "").strip().lower()
        return None if lang in ("", "auto") else lang

    def _fix_words(self, text: str) -> str:
        """转写纠错 —— 按 `asr.fix_words` 表替换听错的专有名词。

        见 `AsrConfig.fix_words` 的说明:**「莫西干」会被听成「墨西哥人」**,
        而且这个错会顺着记忆一直存下去,所以要在**转写出口**就纠正。

        ⚠️ 放在**幻觉闸之后**才对 —— 那几道闸比的是 Whisper 的原始输出,
        先替换会让提示词幻觉检测和媒体残留黑名单对不上。
        """
        for wrong, right in (self.cfg.fix_words or {}).items():
            if wrong and wrong in text:
                text = text.replace(wrong, right)
        return text

    def _local_fallback(self, audio16k: np.ndarray, quick: bool) -> str:
        """云端失败时的兜底 —— **会加载本地 Whisper**(显存就占回去了)。

        只在 `cloud_fallback_local: true` 时才走到这儿。默认关闭,
        因为"省显存"正是换云端的目的。
        """
        model = self._load()
        lang = self._resolve_language()
        segments, info = model.transcribe(
            audio16k, language=lang, beam_size=1 if quick else 5,
            vad_filter=False, condition_on_previous_text=False,
            initial_prompt=(None if (quick or lang is None)
                            else (self.cfg.initial_prompt or None)),
        )
        return "".join(s.text for s in segments).strip()

    def _cloud(self, audio16k: np.ndarray, quick: bool) -> str:
        """云端 ASR(阿里云百炼)。

        为什么用**全模态模型**而不是专门的 ASR 接口:
        🔴 百炼的 OpenAI 兼容接口**没有 `/audio/transcriptions`**(实测 404)。
        音频只能走 chat 接口的 `input_audio` 字段,而那条路目前只有
        omni 系列(如 `qwen3.5-omni-flash`)认 —— 专门的 `fun-asr-flash`
        在这个接口下调不通。

        🔴 **必须是 data URL**(`data:audio/wav;base64,...`),**不能传裸 base64**。
        实测裸 base64 会报「The provided URL does not appear to be valid」——
        它把那个字符串当 URL 解析了。这个坑不踩一次基本想不到。

        实测效果(和音色无关的纯转写):0.4 秒、自带标点、准确率优于本地 large-v3。
        """
        import base64
        import io
        import os
        import wave

        try:
            key_env = getattr(self.cfg, "cloud_api_key_env", "DASHSCOPE_API_KEY")
            key = os.environ.get(key_env)
            if not key:
                self.last_drop = f"没设 {key_env}"
                return ""

            # float32 [-1,1] → 16bit PCM WAV
            pcm = np.clip(audio16k, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(TARGET_RATE)
                w.writeframes(pcm16.tobytes())
            data_url = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()

            from openai import OpenAI

            cli = OpenAI(
                api_key=key,
                base_url=getattr(self.cfg, "cloud_base_url",
                                 "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                timeout=getattr(self.cfg, "cloud_timeout_s", 20.0),
            )
            r = cli.chat.completions.create(
                model=getattr(self.cfg, "cloud_model", "qwen3.5-omni-flash"),
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "input_audio",
                         "input_audio": {"data": data_url, "format": "wav"}},
                        {"type": "text",
                         "text": _cloud_prompt(getattr(self.cfg, "cloud_hint", ""))},
                    ],
                }],
                max_tokens=200,
            )
            text = (r.choices[0].message.content or "").strip()
            # 云端给的是**带标点的完整句子**,和 Whisper 的裸文本略有不同 ——
            # 所以**不套用 Whisper 那几道幻觉闸**(它们是按 Whisper 的输出特征写的,
            # 而且 initial_prompt 那套在云端压根不存在)。
            # 但要挡掉模型在没听清时的客套话(见 _CLOUD_ASR_PROMPT 的说明)。
            if text in _CLOUD_ASR_EMPTY:
                self.last_drop = "云端无人声"
                return ""
            return self._fix_words(text)
        except Exception as exc:
            # ⚠️ **不做"失败退回本地"** —— 那会把本地 Whisper 懒加载进来,
            # 显存又占回去,**省显存这个唯一的目的就没了**。
            # 真想要兜底就把 `asr.backend` 改回 `local`(或者设
            # `cloud_fallback_local: true`,那表示你接受显存被占回去)。
            self.last_drop = f"云端失败:{type(exc).__name__}"
            if getattr(self.cfg, "cloud_fallback_local", False) and not quick:
                try:
                    return self._local_fallback(audio16k, quick)
                except Exception:
                    return ""
            return ""

    def transcribe(self, audio: np.ndarray, src_rate: int = TARGET_RATE,
                   quick: bool = False) -> str:
        """audio 是单声道 float32;src_rate 是它的实际采样率。

        `quick=True` 走**廉价通道**:不做束搜索、不加提示词。给"闭嘴"打断那种
        只需要认出两三个关键词的场景用 —— 它每 0.8 秒就要跑一次,用完整配置
        会把这条线程占满,导致真正该转写的话排不上队(实测拖到 3~8 秒)。
        """
        self.last_drop = ""
        if audio.size == 0:
            self.last_drop = "空音频"
            return ""
        audio = to_16k(audio, src_rate)

        # 第一道闸:先挡静音。VAD 偶尔会漏底噪进来,而 Whisper 对静音的响应是
        # **编内容**而不是返回空 —— 这道闸比后面任何过滤都有效。
        if float(np.sqrt(np.mean(audio ** 2))) < self.SILENCE_RMS:
            self.last_drop = "静音"
            return ""

        # ---- 云 ASR 通道 ----
        # 配置 `backend: cloud` 时走这里,**根本不加载本地 Whisper** ——
        # 那正是省下 2G 显存的关键(本地模型会一直常驻在显卡上)。
        if getattr(self.cfg, "backend", "local") == "cloud":
            return self._cloud(audio, quick)

        model = self._load()
        lang = self._resolve_language()
        segments, info = model.transcribe(
            audio,
            # None = 让 Whisper 自己判断语种。
            # ⚠️ 这里以前是写死的 `self.cfg.language`(配置里是 "zh")——
            #    结果是**英语、日语也按中文解码**,出来的全是音译汉字。
            #    那不是"不太准",是模型压根没在做对应语言。
            language=lang,
            # beam_size 提到 5:劣质音频(比如经 VRChat 低码率 Opus 传过来的
            # 手机语音)上,贪心解码(1)错得厉害,束搜索能明显救回来。
            # 代价是转写慢 1~2 倍 —— 上了 GPU 之后这点开销付得起。
            # 但**打断检测不能这么奢侈**(每 0.8 秒跑一次),所以 quick 走贪心。
            beam_size=1 if quick else 5,
            vad_filter=False,                 # 外面已经切好句了,别切两遍
            condition_on_previous_text=False, # 防止上一句污染下一句(重要)
            # 提示词是把中文钉在简体上用的,但它是**纯中文**的 ——
            # 拿它给英/日音频做偏置,会把解码结果往中文拽,比不给还糟。
            # 所以只在**语言写死**时用;自动识别时宁可不给。
            # (打断检测不在乎繁简,本来就不给,顺带少一份幻觉来源。)
            initial_prompt=(None if (quick or lang is None)
                            else (self.cfg.initial_prompt or None)),
            # 🔑 **热词偏置** —— 见 AsrConfig.hotwords 的说明。
            # 实测用户喊「小莫…」被转写成「**那**…」,唤醒词匹配不上,
            # 表现为"叫了没反应"。这不是同音字错(加变体治不了),
            # 是整词被替换,只有在**解码加偏置**才能救。
            # quick(打断检测)不需要 —— 它只要认「闭嘴」那几个词。
            hotwords=(None if quick else (getattr(self.cfg, "hotwords", "") or None)),
        )
        # 记下这次判定的语种,给 pipeline 打日志用(见 Transcribe.__init__ 说明)。
        self.last_lang = getattr(info, "language", "") or ""
        self.last_lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

        # 第二道闸:逐段看 no_speech_prob。
        # ⚠️ 先落成 list —— segments 是**生成器**,下面判空时还要再遍历一次,
        #    不落list第二次就拿到空了(而且不会报错,只是静默少一批数据)。
        segs = list(segments)
        nsm = float(getattr(self.cfg, "no_speech_max", 0.8))
        parts = [s.text for s in segs if s.no_speech_prob <= nsm]
        text = "".join(parts).strip()
        if quick:
            # quick 走的是捷径,下面几道幻觉闸都不跑,所以这里纠错是安全的
            return self._fix_words(text)
        if not text:
            # 全被判成"不是语音"。**把最高的那个概率留下来** —— 不记数值就
            # 分不清两种完全相反的情况:
            #   0.6~0.7  边缘,很可能是**真话被误杀**(阈值该放宽)
            #   0.85+    确实没人说话,是音乐/噪音(无视即可)
            self.last_nospeech = max((s.no_speech_prob for s in segs), default=0.0)
            self.last_drop = "无人声"
            return ""
        # 第三道闸:挡 initial_prompt 幻觉。**只在真给过提示词时才查** ——
        # 自动识别时提示词根本没送进去,拿它比对是无意义的,还可能误杀
        # 用户正常说的话。
        if lang is not None and self._looks_like_prompt_echo(text):
            self.last_drop = "提示词幻觉"
            return ""
        # 第四道闸:挡媒体署名类残留(「字幕by索兰娅」这种)。
        if self._looks_like_media_artifact(text):
            self.last_drop = "媒体残留"
            return ""
        # 第五道闸:同一句反复出现。**黑名单补不完的那类靠它兜底** ——
        # 它抓的是"行为"而不是"内容",所以不认识的幻觉也能挡住。
        if self._looks_like_repeat(text):
            self.last_drop = "重复幻觉"
            return ""
        # 纠错**放在所有闸门之后** —— 那几道闸比的是 Whisper 的**原始输出**
        # (提示词相似度、媒体残留黑名单都是拿原始文本建的),
        # 先替换会让它们对不上、白白漏掉幻觉。见 `_fix_words`。
        return self._fix_words(text)
