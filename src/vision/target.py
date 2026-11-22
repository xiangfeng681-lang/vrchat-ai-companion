"""从画面里找「某个玩家」——用 OCR 读名牌。

**为什么用 OCR 而不是训练一个检测器**:两条路我都评估过。

  - 训练 YOLO 认名牌:要标一两百张图、搭训练环境、导出 ONNX。
    参考项目(tuckerisapizza / AnyaPiPira)走的就是这条,因为他要的是
    "画面里有没有名牌"这个通用信号。
  - **OCR**:一次同时给出**文字内容和位置** —— 正好是"跟谁"和"在哪"
    这两件事。不用标注、不用训练,模型还小(PP-OCR 中文,几 MB)。
    代价是每帧慢一点(约 2 秒)。

我们要的恰恰是"跟**特定的某个人**",所以 OCR 是对的工具。

**距离怎么估**:不用深度相机 —— **名牌文本框的像素高度就是距离代理**。
人离得越近,名牌在画面上越大。参考项目用的同一个技巧(它日志里
`vision forward blocked reason=height height=73 maxHeight=50`)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Target:
    """画面里找到的目标。坐标都是**像素**,原点在画面左上角。"""

    text: str            # 名牌上的原文
    conf: float          # OCR 置信度
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


def _norm(s: str) -> str:
    """归一化:去掉空格和装饰符号,只留字母数字和汉字。

    OCR 会读出「莫西干9aa3°」这种带尾巴的(°是"在线"圆点),
    也会在名字中间插空格,所以比对前统一抹平。
    """
    return "".join(ch for ch in s if ch.isalnum() or "一" <= ch <= "鿿").lower()


class NameplateFinder:
    """用 OCR 在画面里找名牌。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._ocr = None

    def _ensure(self):
        if self._ocr is not None:
            return
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()

    def warmup(self) -> None:
        self._ensure()

    def find_all(self, image: np.ndarray) -> list[Target]:
        """读出画面里所有文字,返回带框的列表。"""
        self._ensure()
        result, _ = self._ocr(image)
        if not result:
            return []
        out: list[Target] = []
        for box, text, conf in result:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            out.append(Target(
                text=str(text), conf=float(conf),
                left=int(min(xs)), top=int(min(ys)),
                right=int(max(xs)), bottom=int(max(ys)),
            ))
        return out

    def find(self, image: np.ndarray, name: str) -> Target | None:
        """找名字匹配的目标。多个匹配时返回**最大**的那个(通常=最近)。

        `name` 可以是完整名字,也可以是片段 —— 「莫西干」就能匹配到
        「莫西干9aa3°」。片段匹配是刻意的:OCR 对小字、远距离的名字
        经常只读对一部分,要求全等会漏掉。
        """
        want = _norm(name)
        if not want:
            return None
        cands = [t for t in self.find_all(image) if want in _norm(t.text)]
        # 置信度太低的丢掉,免得把背景里的相似文字当成目标
        cands = [t for t in cands if t.conf >= self.cfg.min_conf]
        if not cands:
            return None
        return max(cands, key=lambda t: t.height)
