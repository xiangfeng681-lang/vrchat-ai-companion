"""让小mo 向前走几步 —— 走 VRChat 的 OSC `/input/MoveForward`。

## 为什么单独做个工具

走路走的是 **VRChat 的 OSC 移动输入**,和 pose_server(身体姿态)、
语音管线是**完全不同的通道** —— 那两样不开也能走。
所以它不需要起任何服务,只要 VRChat 在跑、OSC 开着。

## ⚠️ 会"飘",腿不迈

pose_server 推的是**固定站姿**,双脚钉在地上。所以走起来是**平移过去**,
腿不会交替迈步 —— 那是另一个待办(走路步态,难点是没有膝盖追踪器)。
**这个工具只管"位置动了",不管"看起来像不像在走"。**

## 用法

    python tools/walk.py            # 向前走两步(约 0.9 秒)
    python tools/walk.py 4 1.8      # 走 4 步的时长(1.8 秒)
    python tools/walk.py 2 0.9 back # 后退
    python tools/walk.py 2 0.9 straight:left   # 左移
"""
from __future__ import annotations

import sys
import time

from pythonosc.udp_client import SimpleUDPClient

HOST, PORT = "127.0.0.1", 9000
FWD_S = 0.45          # 一步约 0.45 秒(VRChat 步行速度约 3 m/s,一步约 0.7m)

DIRS = {
    "straight": "/input/MoveForward",     # 前进
    "back": "/input/MoveBackward",
    "left": "/input/MoveLeft",
    "right": "/input/MoveRight",
}


def main() -> int:
    a = sys.argv[1:]
    steps = float(a[0]) if a else 2.0
    secs = float(a[1]) if len(a) > 1 else steps * FWD_S
    raw = a[2] if len(a) > 2 else "straight"

    forward = not raw.startswith(("back", "left", "right", "straight:"))
    if raw.startswith("straight:"):
        axis = raw.split(":", 1)[1]        # straight:left → 左移
    else:
        axis = raw
    addr = DIRS.get(axis)
    if addr is None:
        print(f"不认识的走法 {axis!r}。可用: " + " / ".join(DIRS))
        return 1

    c = SimpleUDPClient(HOST, PORT)
    print(f"向 {axis} 走 {steps:g} 步(按住 {secs:.2f} 秒)…")
    c.send_message(addr, True)
    time.sleep(max(0.05, secs))
    c.send_message(addr, False)
    print("停。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
