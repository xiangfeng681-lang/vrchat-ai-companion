"""监听 AnyaDance 驱动的组播回放 —— 验证「我们的包到底有没有被驱动吃进去」。

## 为什么需要它

`pose_server` 发完 UDP 就完事了,**UDP 没有回执** —— 包发出去对不对、
驱动认不认,发的一方完全不知道。直接进 VRChat 用肉眼看,一旦模型不动,
你分不清是:
  * 协议写错了(包根本没被接受)
  * 还是 FBT 没校准(包被接受了但腿不动 —— 这个坑踩过,极有误导性)
  * 还是 UI 还开着,两路推包在互相盖

这个工具把第一层单独切出来:**驱动收下并处理过的指令,会回放到一个组播组**。
能看到回放 = 协议这层是通的,后面再出问题就一定是校准/UI/模型那边。

## 🔴 唯一一个坑:必须绑回环接口

```python
s.bind(("127.0.0.1", 39571))                    # ← 这样
mreq = struct.pack("4s4s", grp, inet_aton("127.0.0.1"))   # ← 和这样
```

`INADDR_ANY` 监听**40 秒一个包都收不到**,换回环立刻收到(实测)。
驱动是 loopback-only 的,组播根本没往物理网卡上发。

## ⚠️ 另一半的坑:驱动只在"指令有变化"时才回放

UI 一直推中立姿态时**没有任何输出** —— 这不是坏了,是去重了。
所以正确用法是**边跑这个、边让某个东西真的动起来**:
`pose_server` 的待机动画一直在变,所以只要它在跑就该有稳定回放。

## 用法

    .venv\\Scripts\\python.exe tools/anya_sniff.py          # 汇总模式(默认)
    .venv\\Scripts\\python.exe tools/anya_sniff.py --raw    # 打印原始 JSON

Ctrl+C 退出。退出时打印一份总结。
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time
from socket import IP_ADD_MEMBERSHIP, IPPROTO_IP, inet_aton

MCAST_GRP = "239.255.39.71"
MCAST_PORT = 39571

REPORT_EVERY = 0.5          # 汇总模式的刷新间隔(秒)
WATCH = ("hmd", "hip", "left_controller", "right_controller")


def open_socket() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # ⚠️ 绑回环,不是 INADDR_ANY —— 见文件头
    s.bind(("127.0.0.1", MCAST_PORT))
    mreq = struct.pack("4s4s", inet_aton(MCAST_GRP), inet_aton("127.0.0.1"))
    s.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)
    s.settimeout(0.5)
    return s


def extract(msg: dict) -> tuple[dict, str, bool]:
    """从回放事件里挖出真正的姿态。返回 (完整 payload 字典, 来源端口, 是否被接受)。

    ⚠️ 回放**不是把我们发的包原样转出来** —— 它包了一层事件:

        {"event":"command_processed", "source":{"port":53765},
         "command":{"accepted":true, "payload":"{...原始 JSON 字符串...}"}}

    真正的设备姿态在 `command.payload` 里,而且是**字符串**,得再解一次。
    直接读 `msg["devices"]` 会永远拿到空 —— 这个坑我自己先踩了一遍
    (表现为"移动量恒为 0.00",看着像驱动没生效,其实是解析错了)。

    来源端口还兼职当指纹用:UI 发的包带 `inputs`(手柄按键)字段,
    我们发的没有 —— 所以两路同时在推时能一眼分开。
    """
    cmd = msg.get("command") or {}
    raw = cmd.get("payload")
    if not raw:
        return {}, "", False
    try:
        payload = json.loads(raw) or {}
    except Exception:
        return {}, "", False
    src = (msg.get("source") or {}).get("port", "?")
    return payload, str(src), bool(cmd.get("accepted"))


def brief(dev: dict) -> str:
    """一个设备一行摘要 —— 位置 + 有没有在动。"""
    p = (dev or {}).get("pose", {}).get("position")
    if not p:
        return "  --  "
    return f"{p[0]:+6.3f} {p[1]:6.3f} {p[2]:+6.3f}"


def main() -> int:
    raw_mode = "--raw" in sys.argv
    s = open_socket()
    print(f"监听组播回放 {MCAST_GRP}:{MCAST_PORT}(绑定回环)")
    print("⚠️ 如果没有输出,先确认:①SteamVR 在跑 ②SteamVR 里驱动已启用")
    print("   ③有东西真的在动 —— 驱动只在指令**有变化**时才回放,静止时是静默的\n")

    n = 0
    first: float | None = None
    last_t = time.monotonic()
    last_n = 0
    rejected = 0
    # **按来源端口分组** —— UI 和我们的包是交错着来的,不分开统计就是一团糊:
    # 看起来"在动",其实动的全是 UI,我们那路可能压根没被接受。
    # ui 标志靠 `inputs` 字段认(只有 UI 会发手柄按键)。
    srcs: dict[str, dict] = {}

    while True:
        try:
            data, _ = s.recvfrom(8192)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break

        n += 1
        if first is None:
            first = time.monotonic()

        try:
            msg = json.loads(data.decode("utf-8", errors="replace"))
        except Exception:
            if raw_mode:
                print(f"[非 JSON] {data[:120]!r}")
            continue

        if raw_mode:
            print(json.dumps(msg, ensure_ascii=False))
            continue

        payload, src, accepted = extract(msg)
        if not payload:
            continue
        if not accepted:
            rejected += 1

        st = srcs.setdefault(src, {"n": 0, "last": {}, "dev": {},
                                   "moved": {k: 0.0 for k in WATCH},
                                   "ui": "inputs" in payload})
        st["n"] += 1
        devices = payload.get("devices") or {}
        st["dev"] = devices
        # 累计每个设备走了多远 —— 静止的包(UI 的中立姿)位移恒为 0
        for k in WATCH:
            p = (devices.get(k) or {}).get("pose", {}).get("position")
            if not p:
                continue
            if k in st["last"]:
                st["moved"][k] += sum(abs(a - b) for a, b in zip(p, st["last"][k]))
            st["last"][k] = list(p)

        now = time.monotonic()
        if now - last_t >= REPORT_EVERY:
            hz = (n - last_n) / (now - last_t)
            print(f"[{now - first:6.1f}s] 累计 {n:6d} 包  {hz:5.1f} Hz  "
                  f"拒绝 {rejected}", flush=True)
            for src_id, s2 in srcs.items():
                tag = "UI" if s2["ui"] else "我们的"
                m = s2["moved"]
                print(f"      源 {src_id:>6s} [{tag:5s}] {s2['n']:5d} 包  "
                      f"移动 头{m['hmd']:6.2f} 胯{m['hip']:6.2f} "
                      f"手{m['left_controller'] + m['right_controller']:6.2f}")
            last_t, last_n = now, n

    print(f"\n退出。共收到 {n} 个包"
          + (f",历时 {time.monotonic() - first:.1f}s" if first else ""))
    if n == 0:
        print("一个包都没收到 —— 检查 SteamVR 是否在跑、驱动是否启用。")
        return 0
    if rejected:
        print(f"⚠️ 有 {rejected} 个包**被驱动拒绝** —— 协议字段可能不对。")
    for src_id, s2 in srcs.items():
        print(f"\n源 {src_id} = {'UI(验证时应关掉)' if s2['ui'] else '我们的 pose_server'}"
              f",共 {s2['n']} 包")
        for k in WATCH:
            if s2["dev"].get(k):
                print(f"  {k:17s} 位移 {s2['moved'][k]:8.2f}   当前 {brief(s2['dev'][k])}")

    ours = [s for s in srcs.values() if not s["ui"]]
    if not ours:
        print("\n⚠️ 只见 UI 的包,没看到我们的 —— pose_server 没在跑,或者发错端口了。")
    elif sum(ours[0]["moved"].values()) < 0.01:
        print("\n⚠️ 我们的包被接受了,但**完全静止** —— 待机动画没在跑?")
    else:
        print("\n✅ 我们的包被驱动接受、且姿态确实在变 —— **协议这层通过**。")
    return 0


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已停止。")
