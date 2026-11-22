"""采样率转换。

这个工程里有两处必须重采样,原因都跟 WASAPI 有关:

  1. **录音**:回环设备强制 48000Hz,而 Whisper 只吃 16000Hz。
  2. **放音**:Edge-TTS 出 24000Hz,而 WASAPI 共享模式只收设备原生率(通常 48000Hz),
     不转就报 `Invalid sample rate [PaErrorCode -9997]`。

优先用 soxr(高质量、快);没装就退化成线性插值 —— 降采样时没有抗混叠,
高频会折回来,对语音影响有限但不理想。
"""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """单声道 float32 重采样。采样率相同则原样返回(不拷贝)。"""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    try:
        import soxr
        return soxr.resample(audio, src_rate, dst_rate).astype(np.float32)
    except ImportError:
        n = max(1, int(round(len(audio) * dst_rate / src_rate)))
        x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)
