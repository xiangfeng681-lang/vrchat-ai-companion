"""给她编的舞 —— 关键帧编排,不需要 Blender / MMD。

## 怎么编的

六个虚拟设备就是六个控制点。把想让她摆的姿势写成**关键帧**,
中间用平滑插值补出来(smoothstep,两端速度为零,不会有机械的匀速感)。

编排:

    站姿 → 左右摇摆 ×2 → 右手举起挥手 → 双手高举弹跳 → 收回站姿

## 两个约束(都是实测踩出来的,别忽略)

1. **没有膝盖追踪器。** 膝盖是 IK 从「胯 + 双脚」反推的 ——
   所以"屈膝"要靠**把胯降下来**,而不是去动一个不存在的膝盖点。
2. **脚踝在 y≈0.20,不是脚底。** 脚部追踪点戴在脚踝上。
   第一版按"脚底"编了 0.06,腿被 IK 拧得很难看。
3. **摆动时脚要留在原地。** 真人不会一边晃一边挪脚 ——
   重心靠**移胯**表现,脚只做几毫米的补偿。

## 用法

    # 先关掉 AnyaDance 的 UI(它在 60Hz 推中立姿态,会盖掉这里)
    .venv\\Scripts\\python.exe tools/dance.py            # 循环跳
    .venv\\Scripts\\python.exe tools/dance.py --once     # 跳一遍就停
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 39570
HZ = 60

# 基准站姿 —— 从 AnyaDance UI 的默认姿态实测来的,别自己编。
# 左 = 负 X,右 = 正 X;脚踝 y≈0.20,胯 y≈1.006。
BASE = {
    "hmd": [0.0, 1.500, 0.0],
    "left_controller": [-0.25, 1.28, -0.40],
    "right_controller": [0.25, 1.28, -0.40],
    "hip": [0.0, 1.006, -0.05],
    "left_foot": [-0.05, 0.200, 0.10],
    "right_foot": [0.05, 0.200, 0.10],
}


def mk(yaw: float = 0.0, **over) -> dict:
    """从基准站姿派生一个姿态。

    `over` 直接写设备名(它们都是合法的 Python 标识符):
        mk(hip=[0, 0.96, -0.05], right_controller=[0.4, 1.8, -0.3])
    `yaw` 是头部的偏航角(度)。
    """
    d = {k: list(v) for k, v in BASE.items()}
    for k, v in over.items():
        if k in d:
            d[k] = [float(x) for x in v]
    return {"pose": d, "yaw": float(yaw)}


# ---- 关键帧编排 ----------------------------------------------------------
# 每项是 (这一帧持续多少秒, 姿态)。播放时在相邻两帧之间平滑插值。
#
# 摇摆:重心靠**移胯**,脚不动。手臂自然跟随。
_SW_L = mk(yaw=-7, hip=[-0.075, 0.995, -0.05], hmd=[-0.035, 1.498, 0.0],
           left_controller=[-0.32, 1.26, -0.38],
           right_controller=[0.16, 1.32, -0.42])
_SW_R = mk(yaw=7, hip=[0.075, 0.995, -0.05], hmd=[0.035, 1.498, 0.0],
           left_controller=[-0.16, 1.32, -0.42],
           right_controller=[0.32, 1.26, -0.38])

# 右手举起(挥手准备)。左手自然垂着。
_WAVE_UP = mk(yaw=5, hip=[0.03, 1.000, -0.05], hmd=[0.02, 1.500, 0.0],
              left_controller=[-0.24, 1.24, -0.38],
              right_controller=[0.42, 1.80, -0.30])
# 挥手:手左右摆。抬得够高才不会打到头。
_WAVE_A = mk(yaw=6, hip=[0.03, 1.000, -0.05], hmd=[0.03, 1.500, 0.0],
             left_controller=[-0.24, 1.24, -0.38],
             right_controller=[0.55, 1.86, -0.28])
_WAVE_B = mk(yaw=4, hip=[0.03, 1.000, -0.05], hmd=[0.01, 1.500, 0.0],
             left_controller=[-0.24, 1.24, -0.38],
             right_controller=[0.30, 1.78, -0.32])

# 双手高举 + 弹跳。胯降下去 = 屈膝(没有膝盖追踪器,只能这么表现)。
_UP_HI = mk(yaw=0, hip=[0.0, 1.030, -0.05], hmd=[0.0, 1.520, 0.0],
            left_controller=[-0.46, 1.92, -0.22],
            right_controller=[0.46, 1.92, -0.22])
_UP_LO = mk(yaw=0, hip=[0.0, 0.945, -0.05], hmd=[0.0, 1.470, 0.0],
            left_controller=[-0.40, 1.74, -0.26],
            right_controller=[0.40, 1.74, -0.26])

TIMELINE: list[tuple[float, dict]] = [
    (1.4, mk()),               # 起手:站好
    (1.0, _SW_L),
    (1.8, _SW_R),              # 摇摆 ×2 个来回
    (1.8, _SW_L),
    (1.8, _SW_R),
    (0.7, mk()),               # 收一下,准备举手
    (0.9, _WAVE_UP),
    (0.32, _WAVE_A),           # 挥手 —— 快节奏小摆
    (0.32, _WAVE_B),
    (0.32, _WAVE_A),
    (0.32, _WAVE_B),
    (0.7, mk()),
    (1.0, _UP_HI),             # 双手高举
    (0.45, _UP_LO),            # 弹跳 ×2
    (0.45, _UP_HI),
    (0.45, _UP_LO),
    (0.45, _UP_HI),
    (1.6, mk()),               # 放下,收回站姿
]

LOOP_S = sum(d for d, _ in TIMELINE)
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def _smoothstep(x: float) -> float:
    """两端速度为零的平滑曲线 —— 用它插值不会有"机械匀速"感。"""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _quat_yaw(deg: float) -> list[float]:
    r = math.radians(deg) / 2
    return [0.0, math.sin(r), 0.0, math.cos(r)]


def pose_at(t: float) -> dict:
    """t 秒(对循环取模)时的六设备姿态。"""
    t = t % LOOP_S
    acc = 0.0
    for i, (dur, _) in enumerate(TIMELINE):
        if t < acc + dur or i == len(TIMELINE) - 1:
            a = TIMELINE[i][1]
            b = TIMELINE[(i + 1) % len(TIMELINE)][1]      # 末帧接回首帧 → 无缝循环
            u = _smoothstep((t - acc) / dur) if dur > 0 else 0.0
            out = {}
            for k in BASE:
                pa, pb = a["pose"][k], b["pose"][k]
                out[k] = [pa[j] + (pb[j] - pa[j]) * u for j in range(3)]
            yaw = a["yaw"] + (b["yaw"] - a["yaw"]) * u
            return {"pose": out, "yaw": yaw}
        acc += dur
    return TIMELINE[0][1]


def send(p: dict) -> None:
    out = {"version": 1, "devices": {}}
    for name, pos in p["pose"].items():
        out["devices"][name] = {
            "valid": True, "connected": True,
            "pose": {"position": [float(v) for v in pos],
                     "rotation_xyzw": _quat_yaw(p["yaw"]) if name == "hmd"
                                      else [0.0, 0.0, 0.0, 1.0]},
        }
    _sock.sendto(json.dumps(out).encode("utf-8"), (HOST, PORT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只跳一遍")
    args = ap.parse_args()

    print(f"跳舞 → {HOST}:{PORT} @ {HZ}Hz")
    print(f"编排 {len(TIMELINE)} 个关键帧,一个循环 {LOOP_S:.1f} 秒"
          f"{'(跳一遍)' if args.once else '(循环)'}")
    print("⚠️ 确保 AnyaDance 的 UI 已关闭。Ctrl+C 退出。\n")

    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        if args.once and t > LOOP_S:
            break
        send(pose_at(t))
        time.sleep(1.0 / HZ)
    print("结束(驱动保持最后一帧)。")
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
