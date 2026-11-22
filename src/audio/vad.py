"""切句器(VAD)—— 判断「玩家说完了没有」。

用短时能量 + 状态机,不依赖任何模型。够用、零延迟、零依赖。

状态流转:
    IDLE ──能量超阈值──> SPEAKING ──静音够久──> 吐出一整句,回到 IDLE
                                    └─说得太久──> 强制截断吐出

「前摇缓冲」(preroll)很关键:能量检测总是比人声晚一点点触发,
不留缓冲会把第一个字吃掉。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..config import VadConfig


def rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


class EnergyVAD:
    def __init__(self, cfg: VadConfig, sample_rate: int, block_size: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.block_size = block_size

        ms_per_block = 1000.0 * block_size / sample_rate
        self._silence_needed = max(1, int(cfg.silence_ms / ms_per_block))
        self._min_speech = max(1, int(cfg.min_speech_ms / ms_per_block))
        self._max_speech = max(1, int(cfg.max_utterance_ms / ms_per_block))
        self._preroll_len = max(0, int(cfg.preroll_ms / ms_per_block))

        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, self._preroll_len))
        self._speech: list[np.ndarray] = []
        self._speaking = False
        self._silence_run = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        """外部把 AI 自己说的话喂进来时,清掉状态防止自激。"""
        self._preroll.clear()
        self._speech.clear()
        self._speaking = False
        self._silence_run = 0

    def push(self, block: np.ndarray) -> np.ndarray | None:
        """喂一个音频块。说完了就返回整句 float32 数组,否则返回 None。"""
        loud = rms(block) >= self.cfg.energy_threshold

        if not self._speaking:
            if loud:
                self._speaking = True
                self._silence_run = 0
                # 把前摇一起算进去
                self._speech = list(self._preroll)
                self._preroll.clear()
                self._speech.append(block)
            else:
                self._preroll.append(block)
            return None

        # ---- 正在说话 ----
        self._speech.append(block)

        if loud:
            self._silence_run = 0
        else:
            self._silence_run += 1

        too_long = len(self._speech) >= self._max_speech
        quiet_enough = self._silence_run >= self._silence_needed

        if not (too_long or quiet_enough):
            return None

        utterance = self._flush()
        # 太短的当噪声(咳嗽、键盘声)丢掉
        if len(utterance) < self._min_speech * self.block_size:
            return None
        return utterance

    def _flush(self) -> np.ndarray:
        data = np.concatenate(self._speech) if self._speech else np.zeros(0, dtype=np.float32)
        self._speech = []
        self._speaking = False
        self._silence_run = 0
        self._preroll.clear()
        return data
