"""跟随某个玩家 —— 用 OCR 找到他的名牌,然后走过去。

    # 先看它"想做什么",不发任何移动指令(安全,默认就是这个)
    python tools/follow.py --name 莫西干

    # 确认逻辑对了,真的让它走
    python tools/follow.py --name 莫西干 --go

    # 只在原地转头找目标,不前进
    python tools/follow.py --name 莫西干 --go --no-approach

原理(三步都在 src/vision/ 里):
  1. 截图      PrintWindow,被遮挡/异屏都能抓
  2. 找名牌    OCR 读出文字 + 框。**框的像素高度当距离代理** ——
               人越近,名牌在画面上越大,不需要深度相机
  3. 移动      OSC /input/*(VRChat 桌面模式收)

⚠️ **这会向 VRChat 发移动指令,属于"自动操控角色"。**
   本项目其他部分刻意不碰 /input/*,这是唯一的例外,而且是你要的。
   默认 dry-run(只打印不发指令),要真动必须显式加 --go。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pythonosc.udp_client import SimpleUDPClient  # noqa: E402

from src.vision import screen                        # noqa: E402
from src.vision.detect import NameplateDetector      # noqa: E402
from src.vision.target import NameplateFinder        # noqa: E402


class _Cfg:
    """给 NameplateFinder 用的小配置。"""
    min_conf = 0.4


class Mover:
    """封装 VRChat 的 /input/* 移动指令。dry_run 时只打印。"""

    def __init__(self, host: str, port: int, dry_run: bool):
        self._c = SimpleUDPClient(host, port)
        self.dry_run = dry_run
        self._moving = False

    def _send(self, addr: str, val) -> None:
        if not self.dry_run:
            self._c.send_message(addr, val)

    def forward(self, on: bool) -> None:
        self._moving = on
        self._send("/input/MoveForward", bool(on))

    def look_pulse(self, right: bool, seconds: float) -> None:
        """转一小格。**这是实测出来的唯一可靠用法。**

        踩过的坑:以为 LookHorizontal 是连续轴,发 -0.3 想慢慢转 ——
        **完全不转**。实测:
            力度 0.3 / 0.5  → 画面纹丝不动
            力度 1.0        → 转,而且**时长决定转多少**
            时长 0.00s      → 被忽略(消息还没处理就归零了)
            时长 0.02s      → 最小有效格
            时长 >=0.15s    → 封顶,再长也差不多
        所以:力度**必须给满**,要转多少靠**时长**控制,而且有个下限
        (0.02s)和上限(0.15s)夹着。
        """
        self._send("/input/LookHorizontal", 1.0 if right else -1.0)
        time.sleep(max(0.015, min(0.08, seconds)))
        self._send("/input/LookHorizontal", 0.0)

    def look_vertical(self, up: bool, seconds: float) -> None:
        """抬头 / 低头一小格。实测 **+1.0 = 抬头**(截图验证过,不是猜的)。

        行为和水平完全一样:力度必须给满、时长决定幅度、有上下限。
        ⚠️ 起步时**必须先抬头** —— VRChat 桌面视角默认可能是俯视的,
        而名牌渲染在角色头顶上方,不抬头就永远看不到近处的人。
        """
        self._send("/input/LookVertical", 1.0 if up else -1.0)
        time.sleep(max(0.015, min(0.08, seconds)))
        self._send("/input/LookVertical", 0.0)

    def stop(self) -> None:
        self._send("/input/MoveForward", False)
        self._send("/input/MoveBackward", False)
        self._send("/input/LookHorizontal", 0.0)
        self._send("/input/LookVertical", 0.0)
        self._moving = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="要跟的名字(可以是片段)")
    ap.add_argument("--go", action="store_true", help="真的发移动指令(默认只演示)")
    ap.add_argument("--no-approach", action="store_true", help="只转头,不前进")
    ap.add_argument("--detector", choices=("yolo", "ocr"), default="yolo",
                    help="yolo=自训模型(34ms/29Hz) | ocr=读文字(250ms/4Hz,但知道是谁)")
    ap.add_argument("--stop-height", type=int, default=35,
                    help="名牌框高到这个像素高度就停下。⚠️ **越小 = 停得越远** ——"
                         "因为框高随距离增大,阈值小意味着还没走近就停。")
    ap.add_argument("--turn-scale", type=float, default=0.05,
                    help="转向脉冲时长系数:偏差 x 这个 = 转多少秒(会被夹在 0.03~0.12)")
    # 死区别设太大 —— 0.15 时它会在偏左/偏右 15% 的地方就认为"对准了",
    # 实测停下时目标普遍偏左 5%~14%,看起来就是"总往一边偏"。0.06 对准得多。
    ap.add_argument("--turn-deadzone", type=float, default=0.06,
                    help="横向偏差小于这个就不转(默认 0.06;越大越容易停偏)")
    ap.add_argument("--v-deadzone", type=float, default=0.45,
                    help="垂直死区:名牌高过/低于画面中线这么多才抬头/低头")
    ap.add_argument("--v-scale", type=float, default=0.05,
                    help="垂直修正幅度系数(同 turn-scale)")
    ap.add_argument("--search-sweep-cycles", type=int, default=4,
                    help="搜索时同一个俯仰角横扫几轮再换角度")
    ap.add_argument("--search-pitch-span", type=int, default=3,
                    help="同一个方向上换几次俯仰角后掉头(来回覆盖)")
    ap.add_argument("--search-v-s", type=float, default=0.04,
                    help="搜索时每次抬头多少秒")
    ap.add_argument("--search-turn-s", type=float, default=0.12,
                    help="看不到目标时每次转多少秒(搜索扫描的步长)")
    ap.add_argument("--step-on", type=float, default=0.12,
                    help="每轮最多走多少秒(越小越慢越稳;实测 0.20 会冲过头)")
    ap.add_argument("--step-off", type=float, default=0.30,
                    help="走完停多久再进入下一轮")
    ap.add_argument("--min-step-ratio", type=float, default=0.12,
                    help="快到时步子缩到最小的比例(0.12=缩到八分之一)")
    ap.add_argument("--stop-hysteresis", type=float, default=0.20,
                    help="停车迟滞:已经停了的话,要矮到 stop_height*(1-这个) 才重新起步")
    ap.add_argument("--lost-near-ratio", type=float, default=0.5,
                    help="丢失前框高超过 stop_height*这个值,就当作「贴太近」而不是走丢")
    ap.add_argument("--lost-near-s", type=float, default=3.0,
                    help="上面那个判断的有效时间窗(秒)")
    ap.add_argument("--search-pulses", type=int, default=3,
                    help="搜索时一轮发几次脉冲(配合上一条决定扫描速度)")
    ap.add_argument("--interval", type=float, default=0.15, help="循环间隔")
    ap.add_argument("--max-seconds", type=float, default=120, help="安全上限,超过自动停")
    ap.add_argument("--title", default="VRChat")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    mode = "真实移动" if args.go else "DRY-RUN(只打印,不发指令)"
    print(f"跟随目标  : {args.name}")
    print(f"模式      : {mode}")
    print(f"停车阈值  : {args.stop_height}px (越小停得越远)   转向系数: {args.turn_scale}")
    print(f"安全上限  : {args.max_seconds:.0f} 秒")
    print()

    if args.detector == "yolo":
        finder = NameplateDetector()
        print("加载 YOLO 模型 …", end="", flush=True)
        finder.warmup()
        print(f" OK  (provider={finder.providers[0]})")
    else:
        finder = NameplateFinder(_Cfg())
        print("加载 OCR …", end="", flush=True)
        finder.warmup()
        print(" OK")

    # 常驻抓图会话。**不能用一次性的 screen.grab()** —— 它每次都要重建 WGC 会话,
    # 实测 0.31s/次,会把 YOLO 从 29Hz 拖到 3Hz。常驻会话实测 55Hz,latest() 免费。
    win = screen.find_window(args.title)
    if win is None:
        print(f"找不到窗口 {args.title!r}")
        return 1
    stream = screen.WindowStream(win.hwnd, client_only=True)
    stream.start()
    print(f"抓图      : {win.width}x{win.height} 常驻会话 (WGC)")

    mover = Mover(args.host, args.port, dry_run=not args.go)
    was_stopped = None
    last_seen = None          # 最近一次看到目标时它的框高
    last_seen_at = 0.0        # 那次是什么时候
    # 搜索的扫描状态:在当前俯仰角横扫 search_sweep_cycles 轮,再换俯仰角
    search_sweep_left = 0            # 当前俯仰角还要横扫几轮
    search_pitch_up = True           # 下一个俯仰角是往上还是往下
    search_pitch_steps = 0           # 这个方向已经走了几格
    deadline = time.monotonic() + args.max_seconds
    last_report = 0.0

    print("开始。Ctrl+C 停止。\n")
    try:
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            img = stream.latest()      # 没有新帧就返回上一帧,不会阻塞
            if img is None:
                print("还没收到第一帧,稍等 …")
                time.sleep(0.2)
                continue
            frame_w = img.shape[1]
            tgt = finder.find(img, args.name)
            now = time.monotonic()

            if tgt is None:
                mover.forward(False)
                # ⚠️ **刚看到过很近的目标、突然看不见了 —— 那是贴太近了,不是走丢了。**
                # 名牌渲染在头顶上方,离得越近它在画面上越靠上,贴到一定程度
                # 就升出画面上沿了。这时候如果去"转着找",一转身反而把目标彻底
                # 甩出视野 —— 实测就是这样丢掉整整 50 秒的。
                # 所以这种情况**什么都不做,原地等着**,对方稍微退开就能重新看到。
                recently_close = (last_seen is not None
                                  and last_seen >= args.stop_height * args.lost_near_ratio
                                  and now - last_seen_at < args.lost_near_s)
                if recently_close:
                    if now - last_report > 1.5:
                        print(f"[{now - (deadline - args.max_seconds):6.1f}s] "
                              f"刚才离得近({last_seen}px)现在看不到 → 原地等")
                        last_report = now
                    time.sleep(args.interval)
                    continue
                # 真的走丢了才转着找。往**固定一侧**转(不要随机),
                # 这样能均匀扫过一圈,不会来回晃。
                # 一轮发多次脉冲 —— OCR 一轮要 1.1 秒,一次只转一点点的话
                # 扫完一圈要一分多钟,实际用起来像卡住了。
                # 搜索要像**扫描**:保持一个俯仰角横扫几轮,再换一个俯仰角横扫。
                # 之前是每轮既抬头又横扫 —— 那不是扫描,是螺旋上升,最后必然
                # 盯着天花板转。用户原话:"别一直抬头,转一下上下看一下再转一下"。
                if search_sweep_left > 0:
                    search_sweep_left -= 1
                else:
                    # 换一个俯仰角,然后开始新的一轮横扫
                    mover.look_vertical(up=search_pitch_up, seconds=args.search_v_s)
                    search_sweep_left = args.search_sweep_cycles - 1
                    search_pitch_steps += 1
                    # 抬几下就掉头改成落几下 —— 来回覆盖,不会跑出合理的俯仰范围
                    if search_pitch_steps >= args.search_pitch_span:
                        search_pitch_up = not search_pitch_up
                        search_pitch_steps = 0
                for _ in range(args.search_pulses):
                    mover.look_pulse(right=True, seconds=args.search_turn_s)
                if now - last_report > 1.5:
                    print(f"[{now - (deadline - args.max_seconds):6.1f}s] 看不到目标 → 转着找")
                    last_report = now
                time.sleep(args.interval)
                continue

            last_seen = tgt.height
            last_seen_at = now

            # 横向偏差:-1 = 完全在左边,+1 = 完全在右边
            offset = (tgt.cx - frame_w / 2) / (frame_w / 2)
            need_turn = abs(offset) > args.turn_deadzone
            close_enough = tgt.height >= args.stop_height

            # **停车要有迟滞,不能一到线就停、一低于线又走。**
            # 实测框高在阈值附近抖 ±3px,没有迟滞的话它会一步一停地蹭,
            # 看起来像在抽搐。所以:进去用严标准,出来用松标准。
            if tgt.height >= args.stop_height:
                stopped = True
            elif tgt.height < args.stop_height * (1.0 - args.stop_hysteresis):
                stopped = False
            else:
                stopped = was_stopped              # 处在中间地带:保持原状态
            was_stopped = stopped
            close_enough = stopped

            if now - last_report > 0.8:
                what = "够了,停" if close_enough else ("往前走" if not need_turn else "先转头")
                print(f"[{now - (deadline - args.max_seconds):6.1f}s] "
                      # 用 .0f 而不是 d —— YOLO 的框高是 float,OCR 是 int,
                      # 两种检测器换着用时统一的格式才能都吃得下
                      f"框高={tgt.height:3.0f}px  横向偏差={offset:+.2f}  → {what}")
                last_report = now

            # ---- 垂直修正:别让名牌跑出画面上沿 ----
            # 名牌渲染在角色头顶上方,**越近它在画面上越靠上**,贴太近就升出画面
            # —— 这正是"走近就丢"的根因。所以看到目标时顺手把它拉回画面中部。
            # 死区给得比较大(默认 0.45):只在中/下位置才纠,不追求"居中",
            # 否则它会一直抬头看你,最后变成对着天花板。
            frame_h = img.shape[0]
            v_off = (tgt.cy - frame_h / 2) / (frame_h / 2)
            if abs(v_off) > args.v_deadzone:
                mover.look_vertical(up=(v_off < 0),
                                    seconds=abs(v_off) * args.v_scale)

            # ⚠️ **转头优先于停。**
            # 踩过的坑:原来先判 close_enough,结果"够近了"就只停不转 ——
            # 它侧着身子对着你旁边僵住 30 秒,而你明明就在画面左边一半的位置。
            # 够近只意味着"别再往前走了",不意味着"不用面对目标"。
            if need_turn:
                mover.forward(False)
                # 偏差越大转得越久,但夹在 [0.03, 0.12] 秒之间 ——
                # 0.03 是"能转一点点",0.12 是接近封顶但还没到,留点余量
                mover.look_pulse(right=(offset > 0),
                                 seconds=abs(offset) * args.turn_scale)
            elif close_enough:
                mover.forward(False)
            elif not args.no_approach:
                # ⚠️ **走一小步就停,不能一直开着前进。**
                # 踩过的坑:前进跨周期保持 → 一轮 OCR 1.1 秒,这 1.1 秒是
                # "盲走",等下一轮看清楚时人已经走过头了。
                # 所以每轮只走 step_on 秒,剩下的时间留给观察。
                #
                # 而且**越近步子越小** —— 像人一样,快到了就放慢。
                # 不然最后一轮还是会一头撞上去。
                remain = max(0.0, 1.0 - tgt.height / args.stop_height)   # 1=很远 0=够近
                burst = args.step_on * (args.min_step_ratio
                                        + (1 - args.min_step_ratio) * remain)
                mover.forward(True)
                time.sleep(burst)
                mover.forward(False)
                time.sleep(args.step_off)
            else:
                mover.forward(False)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    finally:
        mover.stop()
        stream.close()          # 常驻 WGC 会话要显式关掉,不然会一直占着
        print("\n已停止(移动指令已清零)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
