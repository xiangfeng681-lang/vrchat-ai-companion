"""监听 VRChat **发出来**的 OSC —— 用来确认 VRChat 到底有没有在收麦克风。

这是验证「AI 的声音有没有真的进 VRChat」最可靠的手段,比截图看指示灯强:
VRChat 会按麦克风电平驱动 `/avatar/parameters/Voice`(0~1 浮点)和
`/avatar/parameters/Viseme`(口型),这两个值动 = VRChat 确实在采麦克风。

    python tools/osc_monitor.py                # 听到 Ctrl+C
    python tools/osc_monitor.py --seconds 20   # 只听 20 秒
    python tools/osc_monitor.py --all          # 把每条都打出来(默认只打 Voice/Viseme)

前提:VRChat 里 OSC 开着。VRChat 默认往 127.0.0.1:9001 发。

⚠️ Viseme / Voice 是 **avatar 自己的参数** —— 如果你的模型没定义它们,
VRChat 就不会发。这时用 --all 看看有什么别的在动。
"""

import argparse
import sys
import time
from collections import Counter

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

WATCH = ("Voice", "Viseme", "viseme", "voice", "MouthOpen", "Mouth")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001, help="VRChat 的 OSC 发送端口")
    ap.add_argument("--seconds", type=float, default=0, help="听多久后退出;0 = 一直听")
    ap.add_argument("--all", action="store_true", help="打印全部消息,不只 Voice/Viseme")
    args = ap.parse_args()

    seen: Counter[str] = Counter()
    t0 = time.monotonic()

    def stamp() -> str:
        return f"[{time.monotonic() - t0:6.2f}s]"

    def on_any(address: str, *osc_args) -> None:
        seen[address] += 1
        if args.all or any(w in address for w in WATCH):
            vals = ", ".join(repr(a) for a in osc_args)
            print(f"{stamp()} {address} = {vals}", flush=True)

    disp = Dispatcher()
    disp.set_default_handler(on_any)

    try:
        server = BlockingOSCUDPServer((args.host, args.port), disp)
    except OSError as exc:
        print(f"绑不上 {args.host}:{args.port} —— {exc}")
        print("多半是已经有别的 OSC 程序占着这个端口。")
        return 1

    print(f"监听 {args.host}:{args.port} …", end="", flush=True)
    if args.seconds:
        print(f" {args.seconds:.0f} 秒")
        server.timeout = args.seconds
        server.handle_request()
    else:
        print(" (Ctrl+C 退出)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

    print(f"\n--- 收到 {sum(seen.values())} 条,{len(seen)} 个不同地址 ---")
    for addr, n in seen.most_common(25):
        print(f"  {n:5d}  {addr}")
    if not seen:
        print("  一条都没有。检查:VRChat 是否开着?OSC 是否 Enabled?端口是否 9001?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
