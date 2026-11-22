"""跟随服务 —— 常驻进程,听管线的指令去跟着主人走。

## 为什么要是常驻的

`tools/follow.py` 是一次性的 CLI(跑一次跟 120 秒然后退出)。而语音要的是
**随时能叫它开始、随时叫它停** —— 一次性工具做不到,每次重启还要重连抓图会话
(WGC 会话建一次 0.31s,而且 YOLO 要重新预热)。

## 和 pose_server 的关系

两个都是常驻服务,但**推的是不同的东西,不冲突**:
    pose_server    → 39570 姿态(站着做什么动作)
    follow_server  → 9000  `/input/*` 移动(往哪走)
⚠️ 但**两个都要求 AnyaDance 的 UI 关掉** —— UI 会以 60Hz 推中立姿态盖掉我们。

## 指令(UDP 39601)

    follow            开始跟随
    stop              停止(并立刻松开方向键)
    status            回一句现在在不在跟
    name:<片段>       换要跟的名字(默认从 config 的 owner_names 取)

## 和 follow.py 的三处不同

1. **抓 `Headset Window`**,不是 VRChat 窗口 —— VR 模式下 VRChat 桌面窗口
   最小化,`frame_bounds` 返回哨兵值、`grab` 直接卡死(2026-09-18 实测)。
2. **`stop_height` 重标成 55** —— 桌面窗口那边是 35,VR 视角 1920×1080 下
   名牌大 2~3 倍。实测主人"够近"是 58~60px。
3. **新增「主人 10 秒没动就停」** —— 见 `STILL_SECONDS`。

用法:
    .venv\\Scripts\\python.exe tools/follow_server.py [--go]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Windows 中文控制台:重定向时默认会用 GBK 编码,中文和 emoji 都会炸
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.vision import screen as S          # noqa: E402
from follow import Mover                    # noqa: E402  (复用调好的那套 OSC 封装)

CMD_PORT = 39601
HZ = 10                     # 跟随循环频率。**不用更快** —— 人走路慢,
                            # 10Hz 已经比"走一步看一眼"密得多;再快只是白烧 YOLO。

# ---- 参数:全部搬自 follow.py 调好的默认值,只有 stop_height 是重标的 ----
# ⚠️ VR 视角重标(桌面窗口是 35)。**越小停得越远** —— 方向别搞反,记忆里栽过。
#
# 55 → 45:实测**冲过头** —— 名牌框高随距离增大,而每走一步跨得太多,
# 等它判断"够了"时人已经贴脸上了。日志原话:
#     框高= 31px → 往前走
#     框高= 62px → 够了,停          ← 一格跨了 31px
# 用户"够近"的手感实测是 58~60px,所以阈值给到 45 留出冲过头的余量,
# 冲完正好落在 55~60。
STOP_HEIGHT = 45
TURN_DEADZONE = 0.06
TURN_SCALE = 0.05
V_DEADZONE = 0.45
V_SCALE = 0.05
SEARCH_SWEEP_CYCLES = 4
SEARCH_PITCH_SPAN = 3
SEARCH_V_S = 0.04
SEARCH_TURN_S = 0.12
SEARCH_PULSES = 3
STEP_ON = 0.12
# 走完停多久再观察。**这个值就是"多久看一眼"** —— 停得越久,盲走越久,
# 越容易冲过头。桌面窗口那边用 0.30 够,VR 视角下同样距离像素变化更快,
# 0.30 会导致一格跨掉一半距离,所以缩到 0.18。
STEP_OFF = 0.18
MIN_STEP_RATIO = 0.12
STOP_HYSTERESIS = 0.20
LOST_NEAR_RATIO = 0.5
LOST_NEAR_S = 3.0

# 🆕 「主人没动就停」——用户要的。
# 判据不是"绝对不动"(人站着也会有微小的头显抖动),而是**名牌在画面里的
# 归一化位置**在 STILL_SECONDS 秒内没超出 STILL_TOLERANCE。
# 用归一化值而不是像素:画面尺寸会变(窗口缩放/分辨率切换)。
STILL_SECONDS = 10.0
STILL_TOLERANCE = 0.04      # 归一化位置变化小于它 = 没动

_stop = threading.Event()
_running = threading.Event()
_lock = threading.Lock()
_target = ""
_last_cmd = ""


# ---------------------------------------------------------------- 指令通道

def listener() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", CMD_PORT))
    global _target, _last_cmd
    while not _stop.is_set():
        try:
            data, _ = s.recvfrom(1024)
            cmd = data.decode("utf-8", "replace").strip()
        except Exception:
            continue
        if not cmd:
            continue
        low = cmd.lower()
        if low in ("stop", "停", "停下"):
            _running.clear()
            print("  ← stop", flush=True)
        elif low.startswith("follow") or low == "跟随":
            _running.set()
            print(f"  ← follow (目标 {_target or '(未设)'})", flush=True)
        elif low == "status":
            print(f"  ← status: {'跟随中' if _running.is_set() else '待机'} 目标={_target!r}",
                  flush=True)
        elif cmd.startswith("name:"):
            _target = cmd[5:].strip()
            print(f"  ← 目标改成 {_target!r}", flush=True)
        _last_cmd = cmd
    s.close()


# ---------------------------------------------------------------- 主循环

class Follower:
    def __init__(self, det, stream, mover, frame_w, frame_h):
        self.det, self.stream, self.mover = det, stream, mover
        self.w, self.h = frame_w, frame_h
        self.was_stopped = False
        self.last_seen = None
        self.last_seen_at = 0.0
        self.sweep_left = 0
        self.pitch_up = True
        self.pitch_steps = 0
        self.last_report = 0.0
        # 「没动」检测
        self.ref_pos = None            # 上一次"认为他动了"时的归一化位置
        self.still_since = None

    def reset(self) -> None:
        self.ref_pos = None
        self.still_since = None

    def _still(self, offset: float, v_off: float) -> bool:
        """主人是不是"没动"了。返回 True = 超过 STILL_SECONDS 没动,该停。"""
        pos = (offset, v_off)
        now = time.monotonic()
        if self.ref_pos is None:
            self.ref_pos, self.still_since = pos, now
            return False
        moved = max(abs(pos[0] - self.ref_pos[0]), abs(pos[1] - self.ref_pos[1]))
        if moved > STILL_TOLERANCE:
            self.ref_pos, self.still_since = pos, now
            return False
        return (now - self.still_since) >= STILL_SECONDS

    def step(self, name: str) -> None:
        img = self.stream.latest()
        if img is None:
            return
        tgt = self.det.find(img, name)
        now = time.monotonic()

        if tgt is None:
            self.mover.forward(False)
            # ⚠️ 刚看到过很近的目标、突然看不见 —— 那是**贴太近**(名牌升出画面上沿),
            #    不是走丢了。这时候**原地等**,别转着找 —— 一转反而把它甩出视野,
            #    实测曾因此丢掉整整 50 秒。
            recently_close = (self.last_seen is not None
                              and self.last_seen >= STOP_HEIGHT * LOST_NEAR_RATIO
                              and now - self.last_seen_at < LOST_NEAR_S)
            if recently_close:
                return
            # 真丢了才扫描式找:同一俯仰角横扫几轮,再换角度
            if self.sweep_left > 0:
                self.sweep_left -= 1
            else:
                self.mover.look_vertical(up=self.pitch_up, seconds=SEARCH_V_S)
                self.sweep_left = SEARCH_SWEEP_CYCLES - 1
                self.pitch_steps += 1
                if self.pitch_steps >= SEARCH_PITCH_SPAN:
                    self.pitch_up = not self.pitch_up
                    self.pitch_steps = 0
            for _ in range(SEARCH_PULSES):
                self.mover.look_pulse(right=True, seconds=SEARCH_TURN_S)
            return

        self.last_seen, self.last_seen_at = tgt.height, now
        offset = (tgt.cx - self.w / 2) / (self.w / 2)
        v_off = (tgt.cy - self.h / 2) / (self.h / 2)
        need_turn = abs(offset) > TURN_DEADZONE

        # 停车迟滞:框高在阈值附近抖 ±3px,没迟滞会一步一停地蹭
        if tgt.height >= STOP_HEIGHT:
            stopped = True
        elif tgt.height < STOP_HEIGHT * (1.0 - STOP_HYSTERESIS):
            stopped = False
        else:
            stopped = self.was_stopped
        self.was_stopped = stopped

        # 🆕 主人没动就停
        if self._still(offset, v_off):
            print(f"  ⏹ 主人 {STILL_SECONDS:.0f} 秒没动 → 停止跟随", flush=True)
            _running.clear()
            self.mover.forward(False)
            return

        if now - self.last_report > 1.0:
            what = "够了,停" if stopped else ("往前走" if not need_turn else "先转头")
            print(f"  框高={tgt.height:3.0f}px  横向={offset:+.2f}  → {what}", flush=True)
            self.last_report = now

        if abs(v_off) > V_DEADZONE:
            self.mover.look_vertical(up=(v_off < 0), seconds=abs(v_off) * V_SCALE)

        # ⚠️ **转头优先于停**:够近只意味着"别再往前走",不意味着"不用面对目标"。
        # 顺序反了会侧着身子僵在原地。
        if need_turn:
            self.mover.forward(False)
            self.mover.look_pulse(right=(offset > 0), seconds=abs(offset) * TURN_SCALE)
        elif stopped:
            self.mover.forward(False)
        else:
            remain = max(0.0, 1.0 - tgt.height / STOP_HEIGHT)
            burst = STEP_ON * (MIN_STEP_RATIO + (1 - MIN_STEP_RATIO) * remain)
            self.mover.forward(True)
            time.sleep(burst)
            self.mover.forward(False)
            time.sleep(STEP_OFF)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="真的发移动指令(默认只演示)")
    ap.add_argument("--title", default="Headset Window",
                    help="抓哪个窗口 —— ⚠️ VR 模式下必须用 Headset Window")
    ap.add_argument("--name", default="莫西干")
    # 检测阈值。默认 0.35 在**远距离会整个失效** —— 2026-09-21 实测,VR 模式下
    # 名牌缩到 48px 时模型只给 0.042 分,`find()` 直接返回 None,表现为
    # "目标丢了" → 静默扫视 → 永远不往前走。
    # ⚠️ 这个 YOLO 是**单类**的(不认名字),阈值压太低会跟到别人身上,
    #    所以别低于画面里干扰框的分数。实测干扰框 0.023。
    ap.add_argument("--conf", type=float, default=0.03,
                    help="YOLO 置信度阈值(默认 0.03,远距离名牌分数很低)")
    args = ap.parse_args()

    from src.vision.detect import NameplateDetector

    class _DetCfg:
        imgsz = 640          # ⚠️ 模型是固定 640 输入的 ONNX,改这个会直接报错
        conf = args.conf

    det = NameplateDetector(_DetCfg())
    det.warmup()
    print(f"  检测阈值 conf={args.conf}")

    win = S.find_window(args.title)
    if win is None:
        print(f"❌ 找不到窗口 {args.title!r} —— VRChat 和 SteamVR 起了吗?")
        return 1
    stream = S.WindowStream(win.hwnd, client_only=True)
    stream.start()
    # ⚠️ **要等第一帧。** WGC 会话 `start()` 只是把线程拉起来,第一帧要过一会儿
    #    才到 —— 立刻 `latest()` 会拿到 None,报"抓不到帧",看着像窗口找错了。
    t_wait = time.monotonic()
    frame = None
    while time.monotonic() - t_wait < 8.0:
        frame = stream.latest()
        if frame is not None:
            break
        time.sleep(0.15)
    if frame is None:
        print("❌ 8 秒都没等到第一帧 —— VRChat 在跑吗?窗口是不是被关了?")
        stream.close()
        return 1
    h, w = frame.shape[:2]

    mover = Mover("127.0.0.1", 9000, dry_run=not args.go)
    global _target
    _target = args.name

    print(f"跟随服务  抓图 {args.title!r} {w}x{h}")
    print(f"  YOLO 检测器就绪   stop_height={STOP_HEIGHT}(VR 视角重标)   "
          f"没动超 {STILL_SECONDS:.0f}s 就停")
    print(f"  移动指令: {'✅ 真发(--go)' if args.go else '⚠️ 只演示(加 --go 才真动)'}")
    print(f"\n指令监听 127.0.0.1:{CMD_PORT}   (follow / stop / status / name:xxx)")
    print("⚠️ AnyaDance 的 UI 必须关掉,否则它会盖掉这里。待机中…\n")

    threading.Thread(target=listener, daemon=True).start()
    f = Follower(det, stream, mover, w, h)
    t0 = time.monotonic()
    try:
        while not _stop.is_set():
            if _running.is_set():
                if f.ref_pos is None:
                    print(f"▶ 开始跟随 {_target!r}", flush=True)
                f.step(_target)
                time.sleep(1.0 / HZ)
            else:
                f.reset()
                mover.forward(False)      # 待机时确保不再前进
                time.sleep(0.1)
                if time.monotonic() - t0 > 20:
                    t0 = time.monotonic()
                    print("  (待机中)", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        _stop.set()
        mover.stop()          # 🔴 **必须清零** —— 不然 VRChat 的前进键会一直按着
        stream.close()
        print("已停止(移动指令已清零)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
