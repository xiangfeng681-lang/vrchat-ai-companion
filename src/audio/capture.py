"""耳朵:持续采集音频块,丢进队列。

两种后端,按 config 里的 capture_device 写法自动选:

  「loopback」 / 「loopback:设备名片段」
      回环采集 —— 抓「扬声器正在播什么」。桌面模式下 VRChat 音频走扬声器,
      抓这个就能听到玩家,而且**你自己照常听得见游戏**。
      必须用 pyaudiowpatch:sounddevice 自带的 PortAudio 没编译 WASAPI 回环。

  「CABLE Output」 / 索引号 / 其它名字片段
      普通录音设备,走 sounddevice。VB-CABLE 那类虚拟线属于这种。

⚠️ 回调跑在实时音频线程里,**必须**只做拷贝 + 入队。
   任何耗时操作(转写、打印、网络)都会导致丢帧。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

import numpy as np

from . import devices

_LOOPBACK_ALIASES = ("loopback", "回环")

# 自动挑回环时要避开的「虚拟设备」关键词。
#
# ⚠️ 这不是洁癖,是必须的:装 VB-CABLE 时它会把自己设成 Windows 默认播放设备,
# 而「抓默认输出的回环」于是会抓到 CABLE——也就是 **AI 自己的嘴**。
# 那样一跑就自激:它说一句 → 自己听见 → 再回一句 → 无限循环。
_VIRTUAL_DEVICE_HINTS = ("cable", "vb-audio", "voicemeeter", "virtual", "virtual audio")


def _looks_virtual(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _VIRTUAL_DEVICE_HINTS)


def is_loopback_spec(spec) -> bool:
    """判断配置值是不是「要走回环」。支持 "loopback" 和 "loopback:名字"。"""
    if not isinstance(spec, str):
        return False
    head = spec.split(":", 1)[0].strip().lower()
    return head in _LOOPBACK_ALIASES


def _loopback_hint(spec: str) -> str:
    """从 "loopback:HECATE" 里取出 "HECATE";没写就返回空串(表示用默认输出)。"""
    return spec.split(":", 1)[1].strip() if ":" in spec else ""


class Capture:
    """统一的采集接口。queue 里永远是 float32 单声道块。"""

    def __init__(self, device_spec, block_ms: int, preferred_rate: int = 48000):
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.dropped = 0
        self.block_ms = block_ms
        self._backend = ""
        self._stream = None
        self._pa = None
        self.index: int | None = None
        self.channels = 1
        self.sample_rate = preferred_rate
        self.name = "(未打开)"

        # 左右声道的能量历史,用来判断"说话的人在左边还是右边"。
        # 见 `direction()` —— 这是"谁在说话"的替代信号(视觉那条路不稳)。
        self._stereo: deque = deque()
        self._stereo_lock = threading.Lock()
        self._stereo_keep_s = 8.0          # 留 8 秒,够覆盖一句话 + 一点余量

        if is_loopback_spec(device_spec):
            self._init_loopback(_loopback_hint(device_spec))
        else:
            self._init_input(device_spec, preferred_rate)

        self.block_size = max(1, int(self.sample_rate * block_ms / 1000))

    # ---------------- 回环后端 ----------------

    def _init_loopback(self, hint: str) -> None:
        import pyaudiowpatch as pa

        self._backend = "loopback"
        self._pa = pa.PyAudio()
        info = self._find_loopback(hint)
        self.index = info["index"]
        self.channels = max(1, int(info["maxInputChannels"]))
        # 回环只能按设备原生率开,不给你挑
        self.sample_rate = int(info["defaultSampleRate"])
        self.name = f"{info['name']} (回环)"

    def _find_loopback(self, hint: str) -> dict:
        import pyaudiowpatch as pa

        loopbacks = list(self._pa.get_loopback_device_info_generator())
        if not loopbacks:
            raise RuntimeError(
                "没找到任何 WASAPI 回环设备。请确认:Windows 上跑,且系统里有可用的扬声器。"
            )

        if not hint:
            wasapi = self._pa.get_host_api_info_by_type(pa.paWASAPI)
            default_out = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])

            if not _looks_virtual(default_out["name"]):
                for lb in loopbacks:
                    if default_out["name"] in lb["name"]:
                        return lb

            # 默认输出是虚拟设备(最常见:装完 VB-CABLE,它把自己设成了默认),
            # 抓它等于抓 AI 自己的嘴。改挑第一个**物理**输出。
            physical = [lb for lb in loopbacks if not _looks_virtual(lb["name"])]
            if physical:
                print(
                    f"[耳朵] ⚠️ 系统默认输出是虚拟设备({default_out['name']}),"
                    f"抓它会自激。\n"
                    f"       已改用: {physical[0]['name']}\n"
                    f"       建议去 Windows 声音设置把默认播放改回你的耳机,"
                    f"或者在 config.yaml 里写死 "
                    f"capture_device: \"loopback:{physical[0]['name'].split(' (')[0]}\""
                )
                return physical[0]

            raise RuntimeError(
                f"所有回环设备看起来都是虚拟的,没法安全地自动选。"
                f"请在 config.yaml 里显式指定,例如 "
                f"capture_device: \"loopback:你的耳机名\"。\n"
                f"可用设备:\n"
                + "\n".join(f"  [{lb['index']}] {lb['name']}" for lb in loopbacks)
            )

        needle = hint.lower()
        hits = [lb for lb in loopbacks if needle in lb["name"].lower()]
        if not hits:
            names = "\n".join(f"  [{lb['index']}] {lb['name']}" for lb in loopbacks)
            raise RuntimeError(
                f"没找到名字含 {hint!r} 的回环设备。可用的有:\n{names}"
            )
        return hits[0]

    def _pa_callback(self, in_data, frame_count, time_info, status_flags):
        import pyaudiowpatch as pa

        if status_flags:
            self.dropped += 1
        audio = np.frombuffer(in_data, dtype=np.float32)
        if self.channels > 1:
            stereo = audio.reshape(-1, self.channels)
            self._note_stereo(stereo)          # ← 先记方向,再平均
            audio = stereo.mean(axis=1)
        self._enqueue(audio.astype(np.float32, copy=True))
        return (None, pa.paContinue)

    # ---------------- 声源方向(左右) ----------------

    def _note_stereo(self, stereo: np.ndarray) -> None:
        """记一块的左右能量。**不是每块都记** —— 太密没必要,还会撑爆缓冲。

        ⚠️ 顺序很重要:**必须在 `mean(axis=1)` 之前调用** —— 那一行会把左右
        信息彻底抹掉(见下面 `direction()` 的说明)。
        """
        n = int(stereo.shape[0])
        if n == 0:
            return
        l = float(np.sqrt(np.mean(np.square(stereo[:, 0]))))
        r = float(np.sqrt(np.mean(np.square(stereo[:, 1]))))
        now = time.monotonic()
        with self._stereo_lock:
            # 同一时刻只留一条(块很密),用时间戳做粗去重
            if self._stereo and now - self._stereo[-1][0] < 0.02:
                pt, pl, pr = self._stereo.pop()
                l = max(l, pl)
                r = max(r, pr)
            self._stereo.append((now, l, r))
            cutoff = now - self._stereo_keep_s
            while self._stereo and self._stereo[0][0] < cutoff:
                self._stereo.popleft()

    def direction(self, t1: float, t2: float) -> float | None:
        """`[t1, t2]` 时间窗内声源偏哪边。**负数=左,正数=右,0=中间**。

        返回 -1.0~1.0 的相对平衡(按能量加权),拿不到就返回 None。

        🔑 **这是"谁在说话"的替代信号。** 视觉的发光描边判据一直不稳定,
        但 VRChat 的玩家语音是**空间音频**渲染的 —— 人在你左边说话,声音就偏左。
        这个信号**不需要身份、不依赖画面**,而且物理上就是声源方向。

        ⚠️ **只能分左右,分不了前后。** 两声道下,"正前方"和"正后方"产生的
        左右能量完全相同 —— 这是信息论上的限制,不是实现问题。
        前后要靠视觉补(人在不在画面里)。实测:人从左走到右,
        相对差 1.4%(中)→ 23.7%(最右),信号很强。

        ⚠️ 安静块要**按能量加权**,不能一视同仁 —— 那点底噪的左右差纯属随机,
        和真说话的块等权会把结果拉到 0 附近。
        """
        with self._stereo_lock:
            seg = [x for x in self._stereo if t1 <= x[0] <= t2]
        num = den = 0.0
        for _t, l, r in seg:
            e = l + r
            if e < 0.004:                     # 太安静,左右差是噪声
                continue
            num += ((r - l) / e) * e
            den += e
        if den <= 0.0:
            return None
        return max(-1.0, min(1.0, num / den))

    # ---------------- 普通输入后端 ----------------

    def _init_input(self, spec, preferred_rate: int) -> None:
        import sounddevice as sd

        self._backend = "input"
        self.index = devices.resolve(spec, "input")
        if self.index is None:
            raise RuntimeError("没有可用的输入设备")
        info = sd.query_devices(self.index)
        self.channels = min(2, max(1, int(info["max_input_channels"])))
        self.sample_rate = preferred_rate
        self.name = devices.describe(self.index)

    def _sd_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.dropped += 1
        if indata.shape[1] > 1:
            self._note_stereo(indata)          # ← 先记方向,再平均(同 loopback)
            mono = indata.mean(axis=1).astype(np.float32)
        else:
            mono = indata[:, 0].copy()
        self._enqueue(mono)

    # ---------------- 公共 ----------------

    def _enqueue(self, block: np.ndarray) -> None:
        try:
            self.queue.put_nowait(block)
        except queue.Full:
            self.dropped += 1

    def open(self) -> "Capture":
        if self._backend == "loopback":
            import pyaudiowpatch as pa

            # 注意:传了 stream_callback 的 PyAudio 流会**自动开始**跑,
            # 不能再调 .start()(那是 sounddevice 的 API,PyAudio 叫 start_stream)。
            self._stream = self._pa.open(
                format=pa.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.index,
                frames_per_buffer=self.block_size,
                stream_callback=self._pa_callback,
            )
        else:
            import sounddevice as sd

            self._stream = sd.InputStream(
                device=self.index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._sd_callback,
            )
            self._stream.start()
        return self

    def __enter__(self) -> "Capture":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                # PyAudio 和 sounddevice 的停止 API 名字不一样,各调各的
                if self._backend == "loopback":
                    self._stream.stop_stream()
                    self._stream.close()
                else:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            finally:
                self._pa = None

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return
