"""常驻动作播放器 —— 平时跑待机动画,收到指令就做动作。

## 为什么必须常驻

不能"来一个动作就起一个进程":
- 每次起进程有几百毫秒空档,动作会**一顿**
- 更重要的是**待机动画要一直在跑** —— 停一下模型就成雕像了

所以它是一个常驻进程:60Hz 不断推姿态,平时推待机,收到指令就切动作,
动作放完自动切回待机。

## 指令通道

    UDP → 127.0.0.1:39600       纯文本或 JSON 都行

    "挥手"                       直接发动作名
    {"gesture": "点头"}          等价
    {"gesture": "stop"}          清空队列,立刻回待机

动作名和别名见 `gestures.py` 的 `GESTURES` / `ALIASES`。

## 过渡怎么做的

从待机切到动作、以及动作收尾,**两端各做 0.25 秒的姿态混合**(smoothstep)。
不混合的话姿态会"啪"地跳过去 —— 这是"动作很怪"最容易被忽略的一个来源。

## 用法

    # ⚠️ 先关掉 AnyaDance 的 UI!它在 60Hz 推中立姿态,会盖掉这里
    .venv\\Scripts\\python.exe tools/pose_server.py

    然后另开一个终端发指令测试:
    .venv\\Scripts\\python.exe -c "import socket;socket.socket(2,2).sendto('挥手'.encode(),('127.0.0.1',39600))"
"""
from __future__ import annotations

import json
import math
import socket
import sys
import threading
import time
from collections import deque

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import gestures as G            # noqa: E402

ANYA_HOST, ANYA_PORT = "127.0.0.1", 39570
CMD_PORT = 39600
HZ = 60

BLEND_IN = 0.25               # 切入动作的混合时长(秒)
BLEND_OUT = 0.25              # 收回待机的混合时长
QUEUE_MAX = 2                 # 同时最多排几个动作

# ---- 待机动画(和 tools/idle_pose.py 同一套参数)--------------------------
BREATH_HZ, BREATH_M = 0.22, 0.010
SWAY_HZ, SWAY_M = 0.07, 0.020
HEAD_YAW_DEG = 6.0

# ---- 转向(对准说话的人)--------------------------------------------------
# 指令格式:`look:<度数>`,负=左,正=右。由管线在"听到谁在说话"时发过来。
MAX_YAW = 60.0        # 转到头也别超过这个角度 —— 再大身体会拧成麻花
# ⚠️ **必须和 `src/config.py` 的 `behavior.look_hold_s` 一致** ——
# 管线靠这个时长判断"这边是不是已经自动回正了",不一致会累积出错误角度。
# 实测 6 秒太短:说着说着它就转回正面,像心不在焉。
LOOK_HOLD_S = 30.0    # 最后一次转向指令之后,保持这么久再自动回正
LOOK_SPEED = 0.08     # 每帧(60Hz)逼近目标的比例。0.08 ≈ 0.2 秒到位

_anya = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_queue: deque[str] = deque()
_cur: str | None = None
_cur_start = 0.0
_lock = threading.Lock()

_look = 0.0           # 当前偏航(度)—— 平滑值
_look_target = 0.0    # 目标偏航
_look_until = 0.0     # 保持到这个时刻,之后回正

# ---- 俯仰(抬头/低头)----------------------------------------------------
# 为什么需要:**VRChat 的 chat 气泡渲染在名牌正上方**,而人站近时名牌
# 本来就顶到画面上沿了 —— 气泡再往上就被切掉,OCR 只能读到残缺的一条
# (实测"11",其实是更长的一句话)。抬一点头给气泡留出空间才能读全。
#
# ⚠️ **符号是实测标定的,别按坐标系推导。** 我按右手系推出来是"正=低头",
#    实测**正好相反**。用名牌在画面里的高度量出来的:
#        pitch:-10 → 名牌 y 从 127 变到 14(**物体上移 = 镜头低头**)
#        pitch:+10 → 名牌 y 变到 223(**物体下移 = 镜头抬头**)
#    即 **正 = 抬头**。±10° ≈ ±100px(画面高 703),所以想挪 50px 就用 5°。
#    (偏航那边也栽过同样的跟头,见 `rotate_y` 的注释 —— 这个系统里
#     "按直觉推导方向"不可靠,**一律实测**。)
# ⚠️ 俯仰是**持续生效**的(不像转向会回正)—— 它是个常驻的观察角度。
HEAD_PITCH_DEG = 5.0    # 正 = 抬头(给名牌上方的 chat 气泡留余量)
_pitch = HEAD_PITCH_DEG


def idle_at(t: float) -> tuple[dict, float]:
    """待机姿态 + 头部偏航角(度)。和 idle_pose.py 的参数保持一致。"""
    br = math.sin(2 * math.pi * BREATH_HZ * t)
    sw = math.sin(2 * math.pi * SWAY_HZ * t)
    yaw = HEAD_YAW_DEG * (0.7 * math.sin(2 * math.pi * 0.045 * t)
                          + 0.3 * math.sin(2 * math.pi * 0.031 * t + 1.1))
    d = {k: list(v) for k, v in G.BASE.items()}
    d["hip"][0] += SWAY_M * sw
    d["hip"][1] += BREATH_M * br
    d["hmd"][0] += SWAY_M * 0.6 * sw
    d["hmd"][1] += BREATH_M * br
    for h in ("left_controller", "right_controller"):
        d[h][0] += SWAY_M * 0.8 * sw
        d[h][1] += BREATH_M * 0.6 * br
    for f in ("left_foot", "right_foot"):
        d[f][0] += 0.004 * sw
    return d, yaw


def listener() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", CMD_PORT))
    print(f"指令监听 127.0.0.1:{CMD_PORT}")
    while True:
        data, _ = s.recvfrom(4096)
        raw = data.decode("utf-8", errors="replace").strip()
        if not raw:
            continue
        # ---- 转向指令:`look:<度数>`(负=左,正=右)----
        # 和动作共用一个端口,靠前缀区分。管线在"判断出说话人在哪边"之后发。
        if raw.lower().startswith("look:"):
            try:
                deg = float(raw.split(":", 1)[1])
            except Exception:
                continue
            deg = max(-MAX_YAW, min(MAX_YAW, deg))
            with _lock:
                globals()["_look_target"] = deg
                globals()["_look_until"] = time.monotonic() + LOOK_HOLD_S
            print(f"  ← 转向 {deg:+.0f}°")
            continue

        # ---- 俯仰指令:`pitch:<度数>`(**正 = 抬头**,实测标定)----
        # 常驻生效,不会自动回正。调它是为了给名牌上方的 chat 气泡留余量。
        if raw.lower().startswith("pitch:"):
            try:
                deg = float(raw.split(":", 1)[1])
            except Exception:
                continue
            deg = max(-30.0, min(30.0, deg))
            with _lock:
                globals()["_pitch"] = deg
            print(f"  ← 俯仰 {deg:+.1f}° ({'抬头' if deg < 0 else '低头'})")
            continue

        name = raw
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                if "look" in obj:
                    deg = max(-MAX_YAW, min(MAX_YAW, float(obj["look"])))
                    with _lock:
                        globals()["_look_target"] = deg
                        globals()["_look_until"] = time.monotonic() + LOOK_HOLD_S
                    print(f"  ← 转向 {deg:+.0f}°")
                    continue
                name = str(obj.get("gesture", "")).strip()
            except Exception:
                continue
        with _lock:
            if name in ("stop", "停", "停止"):
                _queue.clear()
                globals()["_cur"] = None
                print("  ← 停止,回待机")
                continue
            g = G.resolve(name)
            if not g:
                print(f"  ← {name!r} 不认识(动作库里没有)")
                continue
            if len(_queue) >= QUEUE_MAX:
                _queue.popleft()
            _queue.append(g)
            print(f"  ← {name}  →  {g}  (队列 {len(_queue)})")


def send(pose: dict, head_quat: list[float]) -> None:
    out = {"version": 1, "devices": {}}
    for name, pos in pose.items():
        if name == "hmd":
            rot = head_quat
        elif name in G.BASE_ROT:
            # ⚠️ 手柄朝向必须**跟着手臂方向一起转**。
            # 固定值只有手臂垂下时才对,举起来就会把腕子拧断(踩过)。
            rot = G.controller_rot(name, pos)
        else:
            # hip / 双脚:实测就是单位四元数,不旋转
            rot = G.IDENT
        out["devices"][name] = {
            "valid": True, "connected": True,
            "pose": {"position": [float(v) for v in pos],
                     "rotation_xyzw": [float(v) for v in rot]},
        }
    _anya.sendto(json.dumps(out).encode("utf-8"), (ANYA_HOST, ANYA_PORT))


def main() -> int:
    global _cur, _cur_start
    print(f"动作播放器 → {ANYA_HOST}:{ANYA_PORT} @ {HZ}Hz")
    print(f"动作库 {len(G.GESTURES)} 个: {' / '.join(G.GESTURES)}")
    print("⚠️ 确保 AnyaDance 的 UI 已关闭,否则它会盖掉这里。Ctrl+C 退出。\n")

    threading.Thread(target=listener, daemon=True).start()

    t0 = time.monotonic()
    last_report = 0.0
    while True:
        t = time.monotonic() - t0
        idle_pose, idle_yaw = idle_at(t)

        with _lock:
            if _cur is None and _queue:
                _cur = _queue.popleft()
                _cur_start = t
                print(f"  ▶ 开始动作 {_cur} ({G.duration(_cur):.1f}s)")
            cur = _cur

        # ---- 算出这一帧该用什么姿态和头部偏航 ----
        if cur is None:
            pose, head_yaw = idle_pose, idle_yaw
        else:
            tg = t - _cur_start
            gd = G.duration(cur)
            got = G.gesture_at(cur, tg)
            if got is None:
                with _lock:
                    _cur = None
                pose, head_yaw = idle_pose, idle_yaw
            else:
                gpose, gquat = got
                # 两端各做一段混合 —— 不然姿态会"啪"地跳过去
                w_in = G.smoothstep(tg / BLEND_IN) if BLEND_IN > 0 else 1.0
                w_out = (G.smoothstep((gd - tg) / BLEND_OUT)
                         if BLEND_OUT > 0 else 1.0)
                w = min(w_in, w_out)
                pose = G.lerp_pose(idle_pose, gpose, w)
                head_yaw = idle_yaw + (math.degrees(2 * math.asin(
                    max(-1.0, min(1.0, gquat[1])))) - idle_yaw) * w

        # ---- 转向:平滑逼近目标,超时自动回正 ----
        # 用简单的指数平滑而不是线性插值 —— 转头的起停本来就该有惯性感。
        with _lock:
            tgt = _look_target if time.monotonic() < _look_until else 0.0
        globals()["_look"] = _look + (tgt - _look) * LOOK_SPEED
        if tgt == 0.0 and abs(_look) < 0.05:
            globals()["_look"] = 0.0        # 免得留个 0.03° 永远回不到正
        if _look:
            pose = G.rotate_y(pose, _look)
            head_yaw = head_yaw + _look      # 头跟着一起转,别只扭身子

        # 俯仰常驻生效 —— 给名牌上方的 chat 气泡留余量(见 HEAD_PITCH_DEG)
        send(pose, G.quat(yaw=head_yaw, pitch=_pitch))

        if t - last_report > 20:
            print(f"  [{t:7.1f}s] 心跳", flush=True)
            last_report = t
        time.sleep(1.0 / HZ)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已停止(驱动保持最后一帧)。")
