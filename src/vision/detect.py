"""从画面里找名牌 —— 用自己训练的 YOLO,不是 OCR。

**和 `target.py` 的分工**:
  - `target.py` 用 OCR,能读出**文字内容和位置**。慢(约 250ms),但知道"是谁"。
  - `detect.py` 用 YOLO,只知道"这里有个目标名牌"。快(约 30~60ms),不知道是谁。

**两者的关系是"老师与学生"**:模型是用 OCR 自动标注出来的
(`tools/autolabel.py` —— OCR 读出的框直接当标注,零人工),所以 YOLO 学的是
**一个特定的名牌**(连同它的头像缩略图)。代价是用户换 Avatar 就失效,
但重训是全自动的,约 20 分钟。

**推理不依赖 torch / ultralytics** —— 只吃 ONNX + onnxruntime,走 CPU。
(装了 onnxruntime-gpu 的话会自动用 GPU,但实测 CUDA 起不来时会静默退回 CPU,
所以要**打印实际用的 provider**,别以为配了就一定在跑 GPU。)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = ROOT / "models" / "nameplate" / "nameplate.onnx"


@dataclass
class Box:
    """画面里的一个检测框。坐标是**原图像素**。"""

    conf: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


class NameplateDetector:
    """YOLOv8 的 ONNX 推理。预处理/后处理都按 ultralytics 的约定手写。"""

    def __init__(self, cfg=None, model_path: str | Path | None = None):
        self.cfg = cfg
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self.imgsz = getattr(cfg, "imgsz", 640) if cfg else 640
        self.conf_thres = getattr(cfg, "conf", 0.35) if cfg else 0.35
        self.iou_thres = getattr(cfg, "iou", 0.45) if cfg else 0.45
        self.providers: list[str] = []
        self._sess = None

    @staticmethod
    def _enable_cuda_dlls() -> None:
        """把 pip 装的 CUDA DLL 目录挂进搜索路径。

        ⚠️ 必须在 `import onnxruntime` **之前**调用。不挂的话,onnxruntime 会
        报 `cublasLt64_12.dll is missing` 然后**静默退回 CPU** ——
        不打印实际 provider 的话根本发现不了。
        (和 src/asr.py 里给 ctranslate2 做的是同一件事,原因也一样。)
        """
        import importlib
        import os
        import sys

        if sys.platform != "win32":
            return
        dirs: list[str] = []
        for pkg in ("cublas", "cudnn", "cufft", "cuda_runtime"):
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

    def _ensure(self):
        if self._sess is not None:
            return
        if not self.model_path.exists():
            raise RuntimeError(
                f"找不到模型 {self.model_path} —— 先跑 tools/autolabel.py 标注,"
                "再用 ultralytics 训练并导出 ONNX(见 README/记忆里的步骤)。"
            )
        self._enable_cuda_dlls()
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        avail = ort.get_available_providers()
        want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self._sess = ort.InferenceSession(str(self.model_path), opts, providers=want)
        # 记下**真正生效**的 provider —— 请求了 CUDA 但库缺失时 onnxruntime 会
        # 静默退回 CPU,不打印的话根本发现不了。
        self.providers = self._sess.get_providers()

    def warmup(self) -> None:
        self._ensure()
        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    # ---------------- 推理 ----------------

    @staticmethod
    def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
        """等比缩放 + 灰边填充到 size×size。返回(图, 缩放比, 左填充, 上填充)。"""
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        # 最近邻缩放:自己做比引 cv2 省一个依赖,而且这里的效果足够
        yi = (np.arange(nh) / r).astype(np.int32).clip(0, h - 1)
        xi = (np.arange(nw) / r).astype(np.int32).clip(0, w - 1)
        resized = img[yi][:, xi]
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def detect(self, img: np.ndarray) -> list[Box]:
        """img 是 RGB uint8。返回原图坐标系下的框列表(已做 NMS)。"""
        self._ensure()
        h0, w0 = img.shape[:2]
        canvas, r, pad_x, pad_y = self._letterbox(img, self.imgsz)

        x = canvas.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]

        out = self._sess.run(None, {self._sess.get_inputs()[0].name: x})[0]
        # ultralytics 导出的是 (1, 4+nc, N):前 4 行是 cx,cy,w,h(640 空间像素),
        # 之后每行是一个类别的分数。转成 (N, 4+nc) 好处理。
        pred = out[0].T
        if pred.shape[1] < 5:
            return []
        scores = pred[:, 4:].max(axis=1)
        keep = scores >= self.conf_thres
        if not keep.any():
            return []
        pred, scores = pred[keep], scores[keep]

        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x0, y0 = cx - bw / 2, cy - bh / 2

        order = np.argsort(-scores)
        x0, y0, bw, bh, scores = x0[order], y0[order], bw[order], bh[order], scores[order]

        keep_idx = self._nms(x0, y0, bw, bh, scores)
        boxes: list[Box] = []
        for i in keep_idx:
            # 从 letterbox 空间还原到原图
            lx = (x0[i] - pad_x) / r
            ty = (y0[i] - pad_y) / r
            rx = (x0[i] + bw[i] - pad_x) / r
            by = (y0[i] + bh[i] - pad_y) / r
            boxes.append(Box(
                conf=float(scores[i]),
                left=float(np.clip(lx, 0, w0)),
                top=float(np.clip(ty, 0, h0)),
                right=float(np.clip(rx, 0, w0)),
                bottom=float(np.clip(by, 0, h0)),
            ))
        return boxes

    def _nms(self, x0, y0, bw, bh, scores) -> list[int]:
        """标准 IoU 抑制。输入已按分数降序。"""
        x1, y1 = x0 + bw, y0 + bh
        areas = bw * bh
        keep: list[int] = []
        idx = list(range(len(scores)))
        while idx:
            i = idx.pop(0)
            keep.append(i)
            if not idx:
                break
            xx0 = np.maximum(x0[i], x0[idx])
            yy0 = np.maximum(y0[i], y0[idx])
            xx1 = np.minimum(x1[i], x1[idx])
            yy1 = np.minimum(y1[i], y1[idx])
            iw = np.clip(xx1 - xx0, 0, None)
            ih = np.clip(yy1 - yy0, 0, None)
            inter = iw * ih
            union = areas[i] + areas[idx] - inter
            iou = inter / np.clip(union, 1e-9, None)
            idx = [j for k, j in enumerate(idx) if iou[k] <= self.iou_thres]
        return keep

    def find(self, img: np.ndarray, name: str | None = None) -> Box | None:
        """找最可信的那个框。没有就返回 None。

        `name` 只是为了和 `NameplateFinder.find(图, 名字)` **签名兼容**,
        好让调用方(跟随工具)能两个检测器换着用 —— YOLO 学的是特定名牌,
        本来就不需要按名字筛,这里直接忽略。

        只跟一个人时画面里通常也只有一个目标名牌;真出现多个(比如站在镜子前),
        取**置信度最高**的 —— 比"取最大的"稳,因为远处误检的框有时反而很大。
        """
        boxes = self.detect(img)
        return max(boxes, key=lambda b: b.conf) if boxes else None
