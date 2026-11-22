"""最小 OSC 通路测试 —— 只为验证「Python 能发到 VRChat」这一件事。

只依赖 python-osc,不 import 项目里任何东西,所以工程还没跑起来时也能用:

    pip install python-osc
    python tools/test_osc.py            # 发一条
    python tools/test_osc.py 自定义文字
    python tools/test_osc.py --loop     # 每 3 秒发一条,方便一边调 VRChat 设置一边看

前提:VRChat 里 Action Menu → Options → OSC → Enabled
"""

import sys
import time

from pythonosc.udp_client import SimpleUDPClient

HOST, PORT = "127.0.0.1", 9000


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    loop = "--loop" in sys.argv
    text = args[0] if args else "OSC 测试 ✓ 如果你看到这行,通路是好的"

    client = SimpleUDPClient(HOST, PORT)
    print(f"目标 {HOST}:{PORT}   /chatbox/input")
    print(f"内容 {text!r}")

    n = 0
    while True:
        client.send_message("/chatbox/input", [text, True])
        n += 1
        if not loop:
            break
        print(f"  已发 {n} 条…")
        time.sleep(3)

    print("\n去 VRChat 看头顶聊天框。没有的话:")
    print("  1) Action Menu → Options → OSC 是不是 Enabled")
    print("  2) VRChat 的 OSC 端口是不是 9000(启动参数 --osc= 改过就不是)")
    print("  3) 防火墙拦了 UDP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
