"""采训练数据:连拍 VRChat 画面,存成图片。

用来喂给名牌检测器。要求画面里**尽可能多地出现名牌**,而且要在
不同位置、不同距离、不同背景、不同光照下 —— 模型见得多才学得会。

    python tools/collect_dataset.py                 # 每 1.5 秒一张,存到 dataset/images
    python tools/collect_dataset.py --interval 1.0 --minutes 5
    python tools/collect_dataset.py --out D:/x     # 换目录

**为什么用 PrintWindow 而不是普通截屏**:见 src/vision/screen.py 的注释 ——
窗口被挡住、在别的显示器上,普通截屏都抓不对。

采完之后用 `tools/label_*.py` 标注(或人工标),再交给 ultralytics 训练。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision import screen  # noqa: E402

try:
    from PIL import Image
except ImportError:
    Image = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/images", help="输出目录")
    ap.add_argument("--interval", type=float, default=1.5, help="每张间隔秒数")
    ap.add_argument("--minutes", type=float, default=0, help="采多久;0=一直采到 Ctrl+C")
    ap.add_argument("--title", default="VRChat", help="窗口标题")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start_idx = len(list(out.glob("*.png")))
    print(f"输出目录 {out}  (已有 {start_idx} 张)")

    info = screen.find_window(args.title)
    if info is None:
        print(f"找不到标题为「{args.title}」的窗口。")
        return 1
    print(f"窗口 {info.title!r}  {info.width}x{info.height}  位置 ({info.left},{info.top})")
    if info.width < 800:
        print(f"⚠️  窗口只有 {info.width} 宽 —— 名牌太小,标注误差会很大。"
              "建议把 VRChat 拉到 1280x720 再采。")

    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    n = 0
    saved = 0
    blank = 0
    print("开始采集。建议:在 VRChat 里走动,让别人(或自己)的名牌出现在各种位置和距离。")
    print("Ctrl+C 结束。\n")
    try:
        while True:
            if deadline and time.monotonic() > deadline:
                break
            img = screen.grab(args.title, client_only=True)
            n += 1
            if img is None:
                blank += 1
            else:
                idx = start_idx + saved
                path = out / f"frame_{idx:05d}.png"
                if Image is not None:
                    Image.fromarray(img).save(path)
                else:
                    import numpy as np
                    np.save(path.with_suffix(".npy"), img)
                saved += 1
                if saved % 10 == 0:
                    print(f"  已存 {saved} 张  ({img.shape[1]}x{img.shape[0]})", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    print(f"\n采集结束: 尝试 {n} 次,成功 {saved} 张,失败 {blank} 次")
    print(f"目录: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
