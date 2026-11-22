"""待机动作 —— 让小mo 站着的时候**像个活人**,而不是雕像。

## 为什么需要这个

一旦用程序接管 AnyaDance 的六个虚拟设备,**VRChat 自带的待机动画就被盖掉了** ——
模型会定格在我们最后一帧推的姿态上,比接管之前还死板(接管前至少有呼吸感)。

所以"会做动作"的前提是**先让它活着**。跳舞再花哨,平时是根木头也没用。

## 动作设计

真人站着不动的时候,其实一直在微动。这里叠了几层**频率互不成整数倍**的正弦波
(不成整数倍 = 永远不重复,看起来不像机器人在循环):

| 层 | 频率 | 幅度 | 干什么 |
|---|---|---|---|
| 呼吸 | 0.22 Hz(≈13 次/分) | 胯 ±1cm | 胸腔起伏,整个人微微上下 |
| 重心 | 0.07 Hz | 胯 ±2cm | 慢慢地左右挪重心 |
| 转头 | 0.045 Hz + 0.031 Hz | 偏航 ±6° | 偶尔看看别处,两个频率叠加所以不定时 |
| 手 | 跟随重心 | ±1.5cm | 自然下垂,随身体轻摆 |

**脚基本不动** —— 真人站着不会一直挪脚。只跟着重心做 4mm 的补偿。

## ⚠️ 坐标系(实测,别搞反)

**左 = 负 X,右 = 正 X**。`hmd` 默认 `(0, 1.5, 0)`,原点在脚下地面,单位米。

## 用法

    # 先关掉 AnyaDance 的 UI!它在 60Hz 推中立姿态,会和这里打架
    .venv\\Scripts\\python.exe tools/idle_pose.py
"""
from __future__ import annotations

import json
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gestures as G                  # noqa: E402  基准站姿/朝向的唯一真相

HOST, PORT = "127.0.0.1", 39570      # AnyaDance 驱动
HZ = 60

# ---- 动作幅度(想调氛围就改这里)------------------------------------------
BREATH_HZ = 0.22          # 呼吸频率
BREATH_CM = 0.010         # 呼吸幅度(米)
SWAY_HZ = 0.07            # 重心摆动频率
SWAY_CM = 0.020           # 重心摆动幅度(米)
HEAD_YAW_DEG = 6.0        # 转头幅度(度)
TOE_CM = 0.004            # 脚跟随重心的补偿(米)

# ---- 基准站姿 -----------------------------------------------------------
# 从 `gestures.py` 取 —— **基准站姿只有一处真相**。
#
# 这里原本抄了一份副本,结果两边一起错了(手的真实高度是 0.83,两处都写成
# 1.28,高了 45cm),而且改的时候极容易只改一处。现在物理上不可能不一致了。
#
# ⚠️ 最容易搞错的一点:**追踪器戴在脚踝和腰上,不是脚底和胯部**。
# 按"脚底"编个 y=0.06 会让 IK 拿着"埋在地下的脚踝"去解算,整条腿拧成麻花。
BASE = G.BASE

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def quat_yaw(deg: float) -> list[float]:
    """绕 Y 轴转 deg 度 → 四元数 XYZW。"""
    r = math.radians(deg) / 2
    return [0.0, math.sin(r), 0.0, math.cos(r)]


IDENT = [0.0, 0.0, 0.0, 1.0]


def pose_at(t: float) -> dict:
    """t 秒时刻的六设备姿态。全是正弦叠加,没有随机 —— 可复现、可回放。"""
    br = math.sin(2 * math.pi * BREATH_HZ * t)              # 呼吸
    sw = math.sin(2 * math.pi * SWAY_HZ * t)                # 重心
    # 两个互不成整数倍的频率叠起来 → 转头时机看起来是"随机的"
    yaw = HEAD_YAW_DEG * (0.7 * math.sin(2 * math.pi * 0.045 * t)
                          + 0.3 * math.sin(2 * math.pi * 0.031 * t + 1.1))

    d = {k: list(v) for k, v in BASE.items()}
    # 整个人随呼吸上下 + 随重心左右
    d["hip"][0] += SWAY_CM * sw
    d["hip"][1] += BREATH_CM * br
    # 头跟着身体走,再加自己的转头
    d["hmd"][0] += SWAY_CM * 0.6 * sw
    d["hmd"][1] += BREATH_CM * br
    # 手自然下垂,随身体轻摆
    d["left_controller"][0] += SWAY_CM * 0.8 * sw
    d["left_controller"][1] += BREATH_CM * 0.6 * br
    d["right_controller"][0] += SWAY_CM * 0.8 * sw
    d["right_controller"][1] += BREATH_CM * 0.6 * br
    # 脚基本钉住,只做一点点重心补偿
    d["left_foot"][0] += TOE_CM * sw
    d["right_foot"][0] += TOE_CM * sw

    # 朝向同样从 gestures 取 —— 手柄有实测基准值,不是单位四元数
    rots = {k: G.rot_of(k) for k in BASE}
    rots["hmd"] = quat_yaw(yaw)
    return {k: (d[k], rots[k]) for k in BASE}


def send(devs: dict) -> None:
    out = {"version": 1, "devices": {}}
    for name, (pos, rot) in devs.items():
        out["devices"][name] = {
            "valid": True, "connected": True,
            "pose": {"position": [float(v) for v in pos],
                     "rotation_xyzw": [float(v) for v in rot]},
        }
    _sock.sendto(json.dumps(out).encode("utf-8"), (HOST, PORT))


def main() -> int:
    print(f"待机动作 → {HOST}:{PORT} @ {HZ}Hz")
    print(f"  呼吸 {BREATH_HZ}Hz ±{BREATH_CM*100:.1f}cm   "
          f"重心 {SWAY_HZ}Hz ±{SWAY_CM*100:.1f}cm   转头 ±{HEAD_YAW_DEG:.0f}°")
    print("⚠️ 确保 AnyaDance 的 UI 已关闭,否则它会盖掉这里。Ctrl+C 退出。\n")

    t0 = time.monotonic()
    n = 0
    try:
        while True:
            send(pose_at(time.monotonic() - t0))
            n += 1
            if n % (HZ * 10) == 0:
                print(f"  [{time.monotonic()-t0:6.1f}s] {n} 帧", flush=True)
            time.sleep(1.0 / HZ)
    except KeyboardInterrupt:
        print("\n已停止(驱动会保持最后一帧)。")
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
