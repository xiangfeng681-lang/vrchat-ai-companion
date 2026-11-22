"""抓 VRChat 窗口的画面。

## 三级方案(按优先级)

**① Windows.Graphics.Capture(WGC)** ← **现在的主力**
用 `windows-capture` 包,按 hwnd 抓。**窗口被挡住、在别的屏、最小化都能抓**,
而且支持 flip-model 交换链。2026-09-17 实测:VRChat 被终端完全盖住时,
0.33s 抓到 1276x767 的完整画面。这是目前唯一能用的正路。

**② PrintWindow + PW_RENDERFULLCONTENT** ← 保底,对 VRChat **已经失效**
它曾经是主力(13ms/张,还拿去训过 YOLO)。但 2026-09-17 实测发现
**对 VRChat / Steam / SteamVR 一律返回 0**,对终端窗口却返回 1。

🔴 **根因是 DXGI flip-model 交换链** —— 现代游戏用它呈现,PrintWindow 抓不到。
之前能用、现在不能用,推测与**挂上 AnyaDance 虚拟头显后 VRChat 转入 VR 渲染路径**
有关。这段代码**留作兜底**(对 CPU 合成的普通窗口仍然有效),别删。

**③ CopyFromScreen** —— 只在前台可见时有效,不在本模块用(要抓被遮挡的窗口)。

## 排查用的判据

想知道某窗口能不能被 PrintWindow 抓,直接调一次看返回值:
`PrintWindow(hwnd, hdc_mem, 2)` → 1 能抓 / 0 不能。**别猜,这个值很明确。**

## 依赖

ctypes 直接调 Win32(不依赖 pywin32)+ `windows-capture`(WGC)。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass

import numpy as np

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32

PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
DWMWA_EXTENDED_FRAME_BOUNDS = 9


def frame_bounds(hwnd: int):
    """窗口**真实**的位置和大小。

    🔴 **绝不要用 GetWindowRect。** 2026-09-17 实测它对这个进程撒谎:

        GetWindowRect                    → (30,175)  741x444   ❌
        GetClientRect                    →          726x407   ❌
        DwmGetWindowAttribute(EXTENDED)  → (63,307) 1276x767   ✅
        WGC 实际抓到的帧                  →         1276x767   ✅

    `GetWindowRect` 返回的是**含不可见调整边框**的矩形,而且会受 DPI 虚拟化
    影响(本进程 DPI-unaware 时拿到的是被缩放过的坐标)。**DWM 扩展边框才是
    DWM 真正合成到屏幕上的那块**,和 WGC 抓到的帧严丝合缝。

    踩过的后果:按 GetWindowRect 的坐标去截屏,VRChat 窗口里会截出 Steam ——
    排查了很久才发现坐标系本身就是错的。
    """
    ext = wt.RECT()
    hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wt.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(ext), ctypes.sizeof(ext))
    if hr == 0 and ext.right > ext.left and ext.bottom > ext.top:
        return (ext.left, ext.top, ext.right - ext.left, ext.bottom - ext.top)
    r = wt.RECT()
    if _user32.GetWindowRect(wt.HWND(hwnd), ctypes.byref(r)):
        return (r.left, r.top, r.right - r.left, r.bottom - r.top)
    return None


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
        ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
        ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG),
        ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wt.DWORD * 3)]


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int
    minimized: bool


def find_window(title_part: str = "VRChat") -> WindowInfo | None:
    """按标题找顶层窗口。找不到返回 None。

    ⚠️ **子串匹配会认错窗口。** 实测搜 "VRChat" 时抓到了标题为
    「⠴ 继续 VRChat AI 项目」的**终端窗口** —— 因为标题里恰好含这个词。
    所以这里**精确匹配优先**,精确匹配不到才退而求其次用"以它开头"。
    真要模糊找,请传更具体的片段。
    """
    exact: list[WindowInfo] = []
    prefix: list[WindowInfo] = []
    key = title_part.lower()

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def _cb(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        n = _user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        _user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        low = title.lower()
        hit = None
        if low == key:
            hit = exact
        elif low.startswith(key):
            hit = prefix
        if hit is None:
            return True
        b = frame_bounds(hwnd)      # ⚠️ 不能用 GetWindowRect,会撒谎(见 frame_bounds)
        if b is None:
            return True
        hit.append(WindowInfo(
            hwnd=hwnd, title=title,
            left=b[0], top=b[1], width=b[2], height=b[3],
            minimized=bool(_user32.IsIconic(hwnd)),
        ))
        return True

    _user32.EnumWindows(_cb, 0)
    if exact:
        return exact[0]
    if prefix:
        return prefix[0]
    return None


# ---------------------------------------------------------------- WGC 后端

def _crop_client(hwnd: int, img: np.ndarray) -> np.ndarray:
    """把整窗抓图裁成客户区(去掉标题栏和边框)。

    ⚠️ 所有后端都**永远返回整个窗口**,想要客户区只能抓完自己裁。
    不裁的话 OCR 会把标题栏的 "VRChat" 也读成画面里的文字。

    🔑 **用比例,不用绝对坐标。** `GetWindowRect`/`GetClientRect` 都活在
    "被 DPI 虚拟化过的"那套坐标里,绝对值不可信(实测 741x444 vs 真实 1276x767);
    但**它俩的比值是无量纲的**,乘到 DWM 的真实帧尺寸上就准。
    绝对偏移(ClientToScreen)不能用 —— 它也在虚拟坐标系里。
    """
    win = wt.RECT()
    cli = wt.RECT()
    if not (_user32.GetWindowRect(wt.HWND(hwnd), ctypes.byref(win))
            and _user32.GetClientRect(wt.HWND(hwnd), ctypes.byref(cli))):
        return img
    win_w, win_h = win.right - win.left, win.bottom - win.top
    if win_w <= 0 or win_h <= 0 or cli.right <= 0 or cli.bottom <= 0:
        return img

    h, w = img.shape[:2]
    cw = int(round(w * cli.right / win_w))
    ch = int(round(h * cli.bottom / win_h))
    if cw <= 0 or ch <= 0 or cw > w or ch > h:
        return img
    # 左右边框对称 → 水平居中;标题栏在上方 → 贴底对齐
    ox = (w - cw) // 2
    oy = h - ch
    return np.ascontiguousarray(img[oy:oy + ch, ox:ox + cw])


def _grab_window_wgc(hwnd: int, *, timeout: float = 2.0) -> np.ndarray | None:
    """WGC 抓一帧。失败返回 None(调用方会退回 PrintWindow)。

    ⚠️ WGC **只在画面有变化时才推帧** —— 完全静止的窗口可能一帧都等不到,
    所以必须有超时看门狗,不能死等。
    """
    try:
        from windows_capture import WindowsCapture
    except Exception:
        return None

    import threading

    got: list[np.ndarray] = []
    cap = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=int(hwnd))

    @cap.event
    def on_frame_arrived(frame, capture_control):        # noqa: ANN001
        if not got:
            got.append(np.asarray(frame.frame_buffer).copy())   # BGRA
        capture_control.stop()

    @cap.event
    def on_closed():
        pass

    # 看门狗:WGC 静止不推帧时 start() 会一直阻塞
    timer = threading.Timer(timeout, lambda: None)
    timer.daemon = True
    timer.start()
    try:
        cap.start()
    except Exception:
        return None
    finally:
        timer.cancel()

    if not got:
        return None
    bgra = got[0]
    if bgra.ndim != 3 or bgra.shape[2] < 3:
        return None
    return np.ascontiguousarray(bgra[:, :, 2::-1])


def _grab_window_printwindow(hwnd: int, *, client_only: bool = False) -> np.ndarray | None:
    """旧主力,现为兜底。**对 flip-model 窗口会返回 None**(见模块说明)。"""
    bounds = frame_bounds(hwnd)
    if bounds is None:
        return None
    # ⚠️ PrintWindow **永远渲染整个窗口**(含标题栏和边框),不管你要的是哪块,
    # 所以位图必须按**整窗**尺寸分配,想要客户区只能抓完自己裁。
    # 之前偷懒按客户区尺寸分配,结果内容整体错位,OCR 把标题栏的 "VRChat"
    # 也当成画面里的文字读了出来。
    _, _, w, h = bounds
    if w <= 0 or h <= 0:
        return None

    hdc_win = _user32.GetDC(hwnd)
    hdc_mem = _gdi32.CreateCompatibleDC(hdc_win)
    hbmp = _gdi32.CreateCompatibleBitmap(hdc_win, w, h)
    old = _gdi32.SelectObject(hdc_mem, hbmp)

    ok = _user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)

    info = _BITMAPINFO()
    info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    info.bmiHeader.biWidth = w
    info.bmiHeader.biHeight = -h              # 负数 = 自上而下,省得再翻转
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    info.bmiHeader.biCompression = 0          # BI_RGB

    buf = ctypes.create_string_buffer(w * h * 4)
    got = _gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(info), DIB_RGB_COLORS)

    _gdi32.SelectObject(hdc_mem, old)
    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(hdc_mem)
    _user32.ReleaseDC(hwnd, hdc_win)

    if not ok or got == 0:
        return None

    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    rgb = np.ascontiguousarray(arr[:, :, 2::-1])        # BGRA -> RGB

    return _crop_client(hwnd, rgb) if client_only else rgb


# ---------------------------------------------------------------- 对外接口

def grab_window(hwnd: int, *, client_only: bool = False,
                timeout: float = 2.0) -> np.ndarray | None:
    """抓窗口内容,返回 RGB 的 numpy 数组 (H, W, 3) uint8。失败返回 None。

    **先走 WGC**(能抓被遮挡的 flip-model 窗口),失败再退回 PrintWindow。
    `client_only=True` 去掉标题栏和边框,只要客户区 —— 训练和推理都该用这个,
    免得模型把标题栏也当成画面的一部分。

    ⚠️ WGC **只在画面有变化时才推帧**,完全静止的窗口会等到 `timeout` 才放弃,
    那种情况下会退回 PrintWindow 再试一次。
    """
    img = _grab_window_wgc(hwnd, timeout=timeout)
    if img is None:
        img = _grab_window_printwindow(hwnd, client_only=False)
    if img is None:
        return None
    return _crop_client(hwnd, img) if client_only else img


def grab(title_part: str = "VRChat", *, client_only: bool = True) -> np.ndarray | None:
    """一步到位:找窗口 + 抓画面。找不到窗口返回 None。"""
    info = find_window(title_part)
    if info is None:
        return None
    return grab_window(info.hwnd, client_only=client_only)


class WindowStream:
    """**常驻**的 WGC 会话,持续收帧,`latest()` 取最近一帧。

    ## 为什么需要它

    `grab_window()` 每次调用都**新开一个 WGC 会话**(建会话 + 等驱动推第一帧),
    实测 **0.31s/次**。对 OCR 路径无所谓(OCR 本身 1.1s),但**对 YOLO 路径是灾难**:
    推理只要 34ms,却被抓图拖到 **3Hz**(原本 29Hz)。

    常驻会话把"建会话"的成本摊掉,之后每帧几乎免费(就是一次内存拷贝)。

    ## 用法

        st = WindowStream(hwnd)          # client_only 默认 True
        try:
            while ...:
                img = st.latest()        # 没有新帧时返回**上一帧**(可能是 None)
                ...
        finally:
            st.close()

    ⚠️ WGC **只在画面有变化时推帧**。窗口完全静止时 `latest()` 会一直返回同一帧,
    这不是 bug —— 那一帧确实就是当前内容。但也意味着**不能用"有没有新帧"当心跳**。
    """

    def __init__(self, hwnd: int, *, client_only: bool = True):
        self.hwnd = int(hwnd)
        self.client_only = client_only
        self._cap = None
        self._ctl = None
        self._frame: np.ndarray | None = None
        self._pending: np.ndarray | None = None

        from windows_capture import WindowsCapture

        self._cap = WindowsCapture(cursor_capture=False, draw_border=False,
                                   window_hwnd=self.hwnd)

        @self._cap.event
        def on_frame_arrived(frame, capture_control):        # noqa: ANN001
            # 🔴 **消费端没取走上一帧就直接丢,别拷。**
            #
            # 踩过的坑:第一版无脑 `self._pending = ...copy()`,而 WGC 在画面有变化时
            # **最多推 138Hz**,每帧 1250×703×4 ≈ **3.5 MB** ——
            # 等于**每秒搬运 480MB 内存,而且拷了根本没人看**。
            # 项目里有两个这样的会话(说话人跟踪 + 名牌定位),
            # 合计把管线推到 **512% CPU(约 5 个核)**,
            # 直接拖慢了 Whisper 和 TTS —— 用户反馈"推理和语音变慢了"就是这个。
            #
            # 消费端(`latest()`)要的只是"最新一帧",中间帧丢掉**完全无损**。
            # 这样拷贝频率就被压到消费端的取用频率(几 Hz 而不是上百 Hz)。
            # ⚠️ frame_buffer 只在回调期间有效,该拷的时候还是必须拷。
            if self._pending is None:
                self._pending = np.asarray(frame.frame_buffer).copy()

        @self._cap.event
        def on_closed():
            pass

    def start(self) -> None:
        self._ctl = self._cap.start_free_threaded()

    def latest(self) -> np.ndarray | None:
        """最近一帧 RGB (H,W,3)。**没有新帧就返回上一帧**。"""
        pend, self._pending = self._pending, None
        if pend is not None:
            rgb = np.ascontiguousarray(pend[:, :, 2::-1])     # BGRA -> RGB
            self._frame = _crop_client(self.hwnd, rgb) if self.client_only else rgb
        return self._frame

    def close(self) -> None:
        try:
            if self._ctl is not None:
                self._ctl.stop()
                self._ctl.wait(2000)
        except Exception:
            pass
        self._ctl = None
