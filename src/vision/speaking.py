"""判断画面里谁在说话 —— 靠 VRChat 名牌的**发光描边**。

## 依据(2026-09-17 实测 + 音画同步验证)

**说话时名牌外框出现一圈明亮的发光描边。** 用耳机回环当自动真值做音画对齐,
**84 个时间片里 81 个一致**(3 个不一致全在说话起止的过渡片),见 [[vrchat-speaker-id]]。

⚠️ **描边颜色随好友状态变**,所以判据**不能写死颜色**:

| | 名字文字 | 说话时描边 |
|---|---|---|
| 好友 | 金色 | **金色发光**(色相≈21) |
| 非好友 | 白色 | **青色发光**(色相≈95) |

🔑 **共同点是"比房间亮得多"** —— 房间 V≈178,发光 V>210。所以**只看亮度**。

## 🔴 找名牌也不能按颜色

名牌**底色随玩家变**(实测同帧里紫底饱和 216、灰黑底饱和仅 9,大概率编码信任等级)。
按"暗紫色"找会漏掉灰底的人。**改用明度**(底是暗的 V≈53~103)+ **形状**(横长条 + 实心)。

## 判据参数(实测标定)

    PAD=20, V_MIN=210, GLOW_MIN=0.010

实测分布:**安静恒为精确的 0.000**(环里全是房间,上不去 210);说话 0.019~0.113。
"""
from __future__ import annotations

import re
import threading
import time
from collections import Counter, deque
from difflib import SequenceMatcher

import cv2
import numpy as np

PAD = 20          # 名牌外扩几像素取环

# "亮"的门限。**现场标定过两轮**:
#   房间背景 V 的中位和 90 分位都是 **178~179** → 门限只要高于它就安全。
#   一开始用 210,那是从**离线的青色发光帧**标的;实测现场的金色发光暗得多,
#   210 会把大半个光晕削掉(占比直接减半)。**190 实测量到的信号强近一倍**,
#   而且离背景仍有 11 的余量。
V_MIN = 190

# 环内亮像素占比阈值。实测**安静恒为精确的 0.0000**(环里全是房间),
# 说话最低 0.0063、常见 0.011~0.059。取 0.004 留足余量。
# ⚠️ 后面 `speaker_at()` 还要**按采样点投票**(要 35% 的点都亮才算),
#    所以这里宁可放宽一点,靠投票去伪,而不是靠卡死阈值。
GLOW_MIN = 0.004
LABEL_EVERY = 3.0  # 每隔几秒 OCR 一次给名牌贴名字

_PUNCT = re.compile(r"[\s\.\,\，\。\!\！\?\？\~～\°\|\/\\\-_\(\)\[\]【】]+")


def norm_name(s: str) -> str:
    """名字归一化 —— 用它当**身份 key**。

    ⚠️ 同一张名牌 OCR 每次读出来都可能不一样(实测读出过 `莫西干9aa30`、
    `莫西干9aa3°`、`莫西干 9aa3`)。不归一化的话,同一个人会碎成好几个身份,
    记忆也跟着碎。去掉空格和标点、统一小写是底线。
    """
    return _PUNCT.sub("", (s or "").strip()).lower()


# ---------------------------------------------------------------- 单帧检测

def find_plates(rgb: np.ndarray) -> list[dict]:
    """找出画面里**所有**名牌。返回 [{bbox, glow, hue}, ...]。

    按**明度 + 形状**找,不按颜色(底色随玩家变,见模块说明)。
    """
    h = rgb.shape[0]
    top = np.ascontiguousarray(rgb[: int(h * 0.62)])   # 名牌总在上方
    hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)         # ⚠️ 传进来必须是 RGB
    V = hsv[:, :, 2]

    m = (V < 150).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))  # 填掉文字打的洞

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2000:
            continue
        if not (1.8 < bw / max(1, bh) < 7.0):          # 名牌是横长条
            continue
        if area / float(bw * bh) < 0.55:               # 实心矩形;头发/地面填不满
            continue

        ry1, ry2 = max(0, y - PAD), min(top.shape[0], y + bh + PAD)
        rx1, rx2 = max(0, x - PAD), min(top.shape[1], x + bw + PAD)
        outer = np.ascontiguousarray(top[ry1:ry2, rx1:rx2])
        ring = np.ones(outer.shape[:2], bool)
        ring[y - ry1:y - ry1 + bh, x - rx1:x - rx1 + bw] = False
        if ring.sum() < 100:
            continue
        ohsv = cv2.cvtColor(outer, cv2.COLOR_RGB2HSV)
        V = ohsv[:, :, 2]

        # 🔑 **四边分开算,取最小值。**
        #
        # 坑一(已修):一开始用"整个环里亮像素的占比",结果**名牌旁边站个白发角色**
        # 就把那一边照得雪亮 → 占比恒定在 0.10~0.16、十几秒纹丝不动,看着像"一直在说话"。
        # 真发光是**贴着名牌一圈的轮廓,四边都有**;旁边的亮物体只污染一边。
        # 取四边**最小值** → 一边亮没用,必须四边都亮。自动免疫。
        #
        # 坑二(更隐蔽,也刚修):**采样带必须贴着名牌,不能放在环的最外沿。**
        # 光晕是紧挨名牌边缘的一圈,往外十几像素就淡没了。第一版取的是环的
        # 最外侧 5 像素,离名牌整整 PAD 远,结果 **hue 明明读到金色(说明有亮像素)、
        # 四个边却全是 0.0000** —— 亮的像素在环的内侧,恰好没被采到。
        k = max(3, PAD // 4)
        sides = (ring[PAD - k:PAD, :], ring[-PAD:-PAD + k, :],
                 ring[:, PAD - k:PAD], ring[:, -PAD:-PAD + k])
        vs = (V[PAD - k:PAD, :], V[-PAD:-PAD + k, :],
              V[:, PAD - k:PAD], V[:, -PAD:-PAD + k])
        fr = [float((v[s] > V_MIN).mean()) if s.sum() else 0.0 for v, s in zip(vs, sides)]
        glow = min(fr)

        bright = V > V_MIN
        rh = ohsv[:, :, 0][ring][bright[ring]]
        out.append({
            "bbox": [x, y, bw, bh],
            "glow": glow,
            "sides": [round(f, 4) for f in fr],
            "hue": int(np.median(rh)) if len(rh) > 20 else None,
        })
    return out


def _label(rgb: np.ndarray, plates: list[dict], finder) -> None:
    """给每个名牌贴名字 —— OCR 全帧一次,按中心距离匹配。

    ⚠️ **一个名牌附近可能有好几段文字。** VRChat 会在名字**下面**再渲染一行
    状态文字(实测 `Mu_ya` 名牌下面挂着 `欧鲸鲸`)。第一版是"后来读到的覆盖前面的",
    结果名字被状态行顶掉了 —— 认出来的说话人就变成 `欧鲸鲸` 这个人**根本不存在**。

    ✅ 改成**取面积最大的那一段** —— 名字的字号明显大于状态行。
    """
    try:
        cands: list[tuple[float, dict, float, str]] = []
        for t in finder.find_all(rgb):
            if not (2 <= len(t.text) <= 20) or t.conf <= 0.5:
                continue
            cx, cy = (t.left + t.right) / 2, (t.top + t.bottom) / 2
            area = max(0, t.right - t.left) * max(0, t.bottom - t.top)
            best, bd = None, 1e9
            for p in plates:
                x, y, bw, bh = p["bbox"]
                d = (cx - (x + bw / 2)) ** 2 + (cy - (y + bh / 2)) ** 2
                if d < bd:
                    bd, best = d, p
            if best is not None and bd < 200 ** 2:
                cands.append((area, best, bd, t.text))
        # 面积优先;面积相近时取离名牌中心更近的
        for area, p, bd, text in sorted(cands, key=lambda c: (-c[0], c[2])):
            p.setdefault("name", text)
    except Exception:
        pass          # OCR 挂了不该影响检测


# ---------------------------------------------------------------- 后台跟踪

class PlateTracker:
    """后台定期 OCR,缓存画面里**所有名牌的名字和位置**。

    用途:配合音频的左右方向,回答"**左边那个正在说话的人是谁**"。

    ## 为什么用 OCR 而不是 find_plates

    `find_plates` 按颜色明暗找名牌,**对底色不同的名牌不稳**(实测绿色名牌
    只圈到一部分)。而 OCR 读的是**文字**,不关心名牌底色 —— 而且我们
    本来就要名字。

    ⚠️ OCR 一次要 0.6~1.1 秒,所以**后台低频跑**(默认 2 秒一次),缓存结果。
    """

    def __init__(self, *, title: str = "VRChat", interval_s: float = 2.0):
        self.title = title
        self.interval_s = interval_s
        self.ok = False
        self._plates: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None

    def start(self) -> bool:
        try:
            from src.vision import screen as S

            win = S.find_window(self.title)
            if win is None:
                return False
            self._stream = S.WindowStream(win.hwnd, client_only=True)
            self._stream.start()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self.ok = True
            return True
        except Exception:
            self.close()
            return False

    def _loop(self) -> None:
        from src.vision.target import NameplateFinder

        class _Cfg:
            min_conf = 0.4

        finder = None
        while not self._stop.is_set():
            try:
                if finder is None:
                    finder = NameplateFinder(_Cfg())
                    finder.warmup()
                frame = self._stream.latest() if self._stream else None
                if frame is not None:
                    w = frame.shape[1]
                    found = []
                    for t in finder.find_all(frame):
                        if not (2 <= len(t.text) <= 20) or t.conf <= 0.5:
                            continue
                        # 归一化横向位置:0=最左,1=最右,0.5=画面正中
                        cx = (t.left + t.right) / 2.0 / max(1, w)
                        found.append({
                            "name": norm_name(t.text),
                            "raw": t.text,
                            "cx": round(cx, 4),
                            "cy": int((t.top + t.bottom) / 2),
                            "w": int(t.right - t.left),
                            "h": int(t.bottom - t.top),
                        })
                    if found:
                        with self._lock:
                            self._plates = found
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def plates(self) -> list[dict]:
        with self._lock:
            return list(self._plates)

    def pick(self, side: float, *, deadzone: float = 0.03) -> dict | None:
        """`side` 是音频给的方向(负=左,正=右)。返回那一侧**最可能**的那个名牌。

        选谁:该侧里**框最大**的 —— 越大=离得越近=越可能是正在跟你说话的人。
        (没有更好的判据:画面里同时有好几个人时,声音只告诉我们左右。)

        `side` 在死区内(|side| < deadzone)返回 None —— **宁可不动,
        也别在人家就站你正前方时乱扭头**。
        """
        if side is None or abs(side) < deadzone:
            return None
        with self._lock:
            cands = [p for p in self._plates
                     if (p["cx"] < 0.5) == (side < 0)]
        if not cands:
            return None
        return max(cands, key=lambda p: p["w"] * p["h"])

    def close(self) -> None:
        self.ok = False
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None


class SpeakerTracker:
    """后台盯着画面,记下**什么时刻谁在说话**。

    ## 为什么要记历史而不是只看"现在"

    VAD 切一句话要 2~3 秒(要等尾部静音),而发光是**瞬时**的。
    等话切出来再去问"现在谁在发光",人家可能已经说完了。
    **必须回头查"这句话的时间窗内谁在发光"。** 这是这个功能最容易翻车的地方。

    ## 用法

        tr = SpeakerTracker()
        if tr.start():                 # 找不到窗口就返回 False,**不要当错误**
            ...
            who = tr.speaker_at(t_start, t_end)
        tr.close()

    ⚠️ **所有对外方法都不抛异常** —— 视觉是锦上添花,绝不能让语音管线崩掉。
    """

    def __init__(self, *, title: str = "VRChat", history_s: float = 60.0,
                 interval: float = 0.25, min_votes: float = 0.35):
        self.title = title
        self.history_s = history_s
        self.interval = interval
        self.min_votes = min_votes

        self._hist: deque[tuple[float, frozenset]] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._finder = None
        self._known: dict[str, str] = {}      # 归一化名 -> 规范名(模糊合并用)
        self.ok = False

    # ---- 启动 ----

    def start(self) -> bool:
        try:
            from src.vision import screen as S

            win = S.find_window(self.title)
            if win is None:
                return False
            self._stream = S.WindowStream(win.hwnd, client_only=True)
            self._stream.start()

            from src.vision.target import NameplateFinder

            class _Cfg:
                min_conf = 0.4

            self._finder = NameplateFinder(_Cfg())
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self.ok = True
            return True
        except Exception:
            self.close()
            return False

    def _loop(self) -> None:
        last_label = -99.0
        plates: list[dict] = []
        while not self._stop.is_set():
            try:
                frame = self._stream.latest() if self._stream else None
                if frame is None:
                    time.sleep(0.1)
                    continue
                t = time.monotonic()
                plates = find_plates(frame)
                if t - last_label >= LABEL_EVERY:
                    _label(frame, plates, self._finder)
                    last_label = t

                speaking = set()
                for p in plates:
                    if p["glow"] > GLOW_MIN and p.get("name"):
                        speaking.add(self._canon(p["name"]))
                with self._lock:
                    self._hist.append((t, frozenset(speaking)))
                    cutoff = t - self.history_s
                    while self._hist and self._hist[0][0] < cutoff:
                        self._hist.popleft()
            except Exception:
                pass          # 视觉线程绝不因异常退出
            time.sleep(self.interval)

    def _canon(self, raw: str) -> str:
        """归一化 + 模糊合并,让 OCR 的小抖动不会把一个人变成两个。"""
        n = norm_name(raw)
        if not n:
            return ""
        if n in self._known:
            return self._known[n]
        for k, canon in self._known.items():
            if abs(len(k) - len(n)) <= 3 and SequenceMatcher(None, k, n).ratio() >= 0.82:
                self._known[n] = canon
                return canon
        self._known[n] = n
        return n

    # ---- 查询 ----

    def speaker_at(self, t_start: float, t_end: float) -> str | None:
        """**这段时间窗内**谁在说话。没有/说不准就返回 None。

        用**投票**:窗口内每个采样点都记了"当时谁在发光",谁占的采样点多谁赢。
        占比不过 `min_votes` 就当"认不准",返回 None —— **宁可不知道,也别认错人**。
        """
        try:
            with self._lock:
                samples = [s for (t, s) in self._hist if t_start <= t <= t_end]
            if not samples:
                return None
            votes = Counter()
            for s in samples:
                for nm in s:
                    votes[nm] += 1
            if not votes:
                return None
            name, cnt = votes.most_common(1)[0]
            return name if cnt / len(samples) >= self.min_votes else None
        except Exception:
            return None

    def current(self) -> set[str]:
        """此刻谁在说话(调试用)。"""
        try:
            with self._lock:
                return set(self._hist[-1][1]) if self._hist else set()
        except Exception:
            return set()

    def close(self) -> None:
        self.ok = False
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None
