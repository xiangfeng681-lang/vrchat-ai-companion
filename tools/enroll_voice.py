"""注册主人声纹 —— 把一段"主人说话"的录音变成 `owner.npy`。

## 录什么

**从实际链路录** —— 也就是说,用 `_abtest/rec_ear.py` 从耳朵那条线录,
而不是拿手机或系统录音机在别处录。理由:运行时听到的就是那条线(经过
VRChat 的 Opus 压缩 + 电量归一),**注册和判定的条件一致才准**。
拿干净录音去注册,上线时分数会整体偏低。

**录多长**:30~60 秒就够,而且**内容要杂**(不同句子、不同语气、有长有短)——
只录一句话会让参考向量过拟合到那句话的音高上。

## 用法

    # 录一段(边录边念)
    .venv\\Scripts\\python.exe _abtest\\rec_ear.py 45 owner_enroll

    # 注册
    .venv\\Scripts\\python.exe tools\\enroll_voice.py _abtest\\owner_enroll.wav

注册完把 `config.yaml` 里的 `voiceprint.enabled` 改成 `true`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.voiceprint import Voiceprint   # noqa: E402
from src.config import load                   # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    wav = Path(sys.argv[1])
    if not wav.exists():
        wav = Path(__file__).resolve().parent.parent / sys.argv[1]
    if not wav.exists():
        print(f"❌ 找不到 {wav}")
        return 1

    cfg = load()
    vp = Voiceprint(cfg.voiceprint)

    a, sr = sf.read(str(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    print(f"音频: {wav}")
    print(f"  {len(a)/sr:.1f}s @ {sr}Hz   峰值 {np.abs(a).max():.3f}")

    n = vp.enroll(a, sr)
    if n == 0:
        print("❌ 没找到有效的声音窗(全程静音?)—— 没注册")
        return 1

    print(f"\n✅ 用 {n} 个窗注册完成")
    print(f"   参考向量 → {vp.root / cfg.voiceprint.profile}")

    # 自检:把刚注册的音频再比一遍。**这不是"准确率"**(拿自己比自己必然高),
    # 只用来确认"流程通了、分数在合理区间"。
    s = vp.score(a, sr)
    print(f"\n自检(同一段音频,分数必然偏高,只看流程通不通):")
    print(f"   中位分数 {s:.4f}   逐窗: {vp.last_detail}")

    print(f"\n下一步:把 config.yaml 里 voiceprint.enabled 改成 true")
    print(f"(阈值 {cfg.voiceprint.threshold} —— 主人最低 0.479 / 其他人最高 0.360)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
