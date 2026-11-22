"""自动标注:让 OCR 给 YOLO 当老师。

    python tools/autolabel.py --name 莫西干            # 生成标注
    python tools/autolabel.py --name 莫西干 --draw 12  # 顺便画出来抽查

**为什么这么做**:
手工画框标几百张图是这件事里最累的部分。但 OCR 本来就能读出文字**和它的框** ——
那正好就是我们要标的东西。于是:

    OCR 读图 → 取文字匹配目标的框 → 直接写成 YOLO 标注

**零人工标注**。而且这个老师和学生是互补的:OCR 慢(约 250ms)但能读字,
YOLO 快(约 50ms)但只会找框 —— 训完之后运行时**根本不需要 OCR**。

副作用是:模型学的是**这一个特定名牌**(带它的头像缩略图)。用户换 Avatar
会失效 —— 但重新跑一遍这个脚本就能重训,**成本几乎为零**,所以可以接受。

没匹配到目标的图**也保留**(标注留空),它们是有价值的负样本:
教模型"没有目标时长什么样",能显著降低误检。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.target import NameplateFinder, _norm   # noqa: E402


class _Cfg:
    min_conf = 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="要标注的目标名(片段即可)")
    ap.add_argument("--images", default="dataset/images")
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--draw", type=int, default=0, help="画 N 张带框的图用于抽查")
    ap.add_argument("--draw-out", default="dataset/sheets")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(img_dir.glob("*.png"))
    if not files:
        print(f"{img_dir} 里没有图片")
        return 1

    from PIL import Image

    finder = NameplateFinder(_Cfg())
    print("加载 OCR …", flush=True)
    finder.warmup()

    want = _norm(args.name)
    hit = miss = 0
    print(f"标注 {len(files)} 张,目标={args.name!r}\n")

    for i, f in enumerate(files, 1):
        arr = __import__("numpy").asarray(Image.open(f).convert("RGB"))
        h, w = arr.shape[:2]
        lines: list[str] = []
        for t in finder.find_all(arr):
            if want not in _norm(t.text) or t.conf < _Cfg.min_conf:
                continue
            # YOLO 格式:class cx cy w h,全部归一化到 0~1
            cx = (t.left + t.right) / 2 / w
            cy = (t.top + t.bottom) / 2 / h
            bw = (t.right - t.left) / w
            bh = (t.bottom - t.top) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if lines:
            hit += 1
        else:
            miss += 1
        (lbl_dir / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if i % 25 == 0:
            print(f"  {i}/{len(files)}   命中 {hit}  未命中 {miss}", flush=True)

    print(f"\n标注完成: 命中 {hit} 张 / 没找到 {miss} 张 (共 {len(files)})")
    print(f"标注目录: {lbl_dir}")
    print(f"两者的比例很重要 —— 全都没命中的话说明截图里压根没出现过目标,得重采。")

    if args.draw:
        out = Path(args.draw_out)
        out.mkdir(parents=True, exist_ok=True)
        pool = [f for f in files if (lbl_dir / f"{f.stem}.txt").read_text(encoding="utf-8").strip()]
        random.seed(0)
        sample = random.sample(pool, min(args.draw, len(pool)))
        for f in sample:
            im = Image.open(f).convert("RGB")
            lines = (lbl_dir / f"{f.stem}.txt").read_text(encoding="utf-8").split()
            from PIL import ImageDraw
            d = ImageDraw.Draw(im)
            for j in range(0, len(lines), 5):
                _, cx, cy, bw, bh = (float(x) for x in lines[j:j + 5])
                x0, y0 = (cx - bw / 2) * im.width, (cy - bh / 2) * im.height
                x1, y1 = (cx + bw / 2) * im.width, (cy + bh / 2) * im.height
                d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
            im.resize((im.width * 2, im.height * 2), Image.LANCZOS).save(out / f"chk_{f.name}")
        print(f"抽查图 {len(sample)} 张 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
