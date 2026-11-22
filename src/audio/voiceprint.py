"""声纹 —— 判断「这句话是不是主人说的」。

## 为什么用声纹,而不是视觉的发光描边

两条路是**互补**的,不是二选一:

| | 视觉发光描边 | 声纹 |
|---|---|---|
| 知道"画面里谁在说" | ✅ | ❌ 只认主人 |
| **背对着 / 不在画面里也认得出** | ❌ | ✅ |
| 依赖 | 得开 `speaker_track`(每 3 秒一次 OCR),且只在画面内 | 纯 CPU,不占显存 |
| 认错人的后果 | 可能把别人的话当成主人的 | 只可能"认不出主人",不会"把别人认成主人"* |

\\* 阈值定得当时。**认不出 = 当非主人**,是安全的那一侧。

## 模型:CAM++(3D-Speaker,阿里)

- **中文语料训练**,192 维向量,27MB,**纯 CPU**(`onnxruntime`)
- 和项目里 YOLO、BGE 记忆是同一套设施,不多一个重量级依赖

## 实测区分度(2026-09-18)

素材:主人 45s 实际链路录音 vs 真实 VRChat 多人对话 18s。

    正样本(主人,留一法) 36 窗   均值 0.7788   最低 0.4794
    负样本(其他人)       14 窗   均值 0.2595   最高 0.3604
    → **零重叠,中间空档 0.119**,阈值取中间任意值都行

⚠️ **负样本只有一段、约 2 个人** —— 阈值 0.42 是在这一批上选的,人多起来要重验。

## 用法

    vp = Voiceprint(cfg.voiceprint)
    if vp.ready:                       # 模型和注册向量都在才为真
        if vp.is_owner(audio, rate):   # audio 是 float32 [-1,1] 单声道
            ...
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# 这些常量跟着 `_abtest/voiceprint_test.py` 的实测走,改一个就得重跑那个 A/B。
SR = 16000          # CAM++ 要 16k
WIN = 2.0           # 窗长(秒)
HOP = 1.0           # 步长(秒)
RMS_MIN = 0.02      # 窗内 RMS 低于它当静音,不算


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """加窗 sinc 低通 + 抽取。

    ⚠️ **不能直接隔点取** —— 48k→16k 直接每 3 个取 1 个,8kHz 以上的能量会
    折叠回可听带内变成噪声,声纹会被污染。(没有 scipy,所以手写。)
    """
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    fc = 0.5 / max(up, down)
    t = np.arange(-64, 65)
    h = np.sinc(2 * fc * t) * np.blackman(len(t))
    h /= h.sum()
    y = np.convolve(x, h * up, mode="same") if up > 1 else np.convolve(x, h, mode="same")
    idx = np.arange(0, len(y), down)
    idx = idx[idx < len(y)]
    return y[idx].astype(np.float32)


def windows(a: np.ndarray) -> list[np.ndarray]:
    """切成有声音的窗。静音窗直接丢 —— 拿噪声去算声纹只会污染参考向量。"""
    n, step = int(WIN * SR), int(HOP * SR)
    out = []
    for s in range(0, len(a) - n + 1, step):
        w = a[s:s + n]
        if float(np.sqrt((w ** 2).mean())) >= RMS_MIN:
            out.append(w)
    return out


class Voiceprint:
    """主人的声纹。**所有对外方法都不抛异常** —— 声纹是锦上添花,绝不能让语音管线崩。"""

    def __init__(self, cfg, *, root: Path | None = None):
        self.cfg = cfg
        self.root = Path(root) if root else Path(__file__).resolve().parent.parent.parent
        self._ex = None
        self._ref: np.ndarray | None = None
        self.last_score: float | None = None     # 调试用:最近一次比对的分数
        self.last_detail: str = ""

    # ---- 加载 ----

    def _extractor(self):
        if self._ex is None:
            import sherpa_onnx
            model = self.root / self.cfg.model
            if not model.exists():
                raise FileNotFoundError(model)
            self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(model), num_threads=2, provider="cpu")
            )
        return self._ex

    @property
    def ready(self) -> bool:
        """模型和注册向量都就位才算可用。任何一步失败都返回 False,不抛。"""
        try:
            self._extractor()
            if self._ref is None:
                p = self.root / self.cfg.profile
                if not p.exists():
                    return False
                v = np.load(str(p)).astype(np.float32)
                self._ref = v / (np.linalg.norm(v) or 1.0)
            return True
        except Exception:
            return False

    # ---- 基础 ----

    def embed(self, w: np.ndarray) -> np.ndarray:
        ex = self._extractor()
        st = ex.create_stream()
        st.accept_waveform(SR, np.ascontiguousarray(w, dtype=np.float32))
        st.input_finished()
        v = np.array(ex.compute(st), dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def enroll(self, audio: np.ndarray, sr: int) -> int:
        """用一段音频注册。返回用到的窗数(0 = 失败)。"""
        a = resample(np.asarray(audio, np.float32), sr, SR)
        ws = windows(a)
        if not ws:
            return 0
        e = np.stack([self.embed(w) for w in ws])
        ref = e.mean(axis=0)
        self._ref = ref / (np.linalg.norm(ref) or 1.0)
        p = self.root / self.cfg.profile
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self._ref)
        return len(ws)

    # ---- 判定 ----

    def score(self, audio: np.ndarray, sr: int) -> float | None:
        """整段音频相对主人声纹的分数。没声音/没注册返回 None。

        ## 🔴 用「最好的两窗平均」,不用中位数

        踩过的坑:一开始取**中位数**,理由是"一段里可能混着别人,中位数抗极端值"。
        **反了** —— 回环里是**混音**(别人的声音 + 你的声音都在),所以一段话里
        "你的窗"和"别人的窗"同时存在时:

            取中位数 → 被别人的窗拉到中间 → **把主人判成陌生人**
            (实测:8.3s 的「哎呦哎呦哎呦,小莫向左走两步」得 0.381,阈值 0.42)

        正确的问法是「**这段里有没有哪个窗明确是你**」,不是「平均下来像不像你」。
        所以取**分数最高的两窗的均值**:

            * 全是主人 → 两窗都高 → 高
            * 主人+别人混合 → 主人的窗拉上来 → 高（**这是关键**）
            * 全是别人 → 没有高窗 → 低

        取**两窗**而不是单窗最高,是为了不让一个偶然的噪声窗说了算。
        """
        if not self.ready:
            return None
        try:
            a = resample(np.asarray(audio, np.float32), sr, SR)
            ws = windows(a)
            if not ws:
                return None
            sc = sorted((float(self.embed(w) @ self._ref) for w in ws), reverse=True)
            self.last_detail = " ".join(f"{s:.2f}" for s in sc)
            top = sc[:2] if len(sc) >= 2 else sc
            return float(sum(top) / len(top))
        except Exception:
            return None

    def is_owner(self, audio: np.ndarray, sr: int) -> bool:
        """是不是主人在说。**认不出/出错一律当"不是"** —— 安全的那一侧。"""
        s = self.score(audio, sr)
        self.last_score = s
        if s is None:
            return False
        return s >= float(self.cfg.threshold)
