"""把 `mmd_solve.py` 解算出来的舞蹈,**直接播到 AnyaDance 驱动上**。

## 为什么要"相对基准",不直接套绝对坐标

两边的**绝对坐标对不上**,而且不是差一个常数:

| | MMD 解算出来的 | 我们的基准(`gestures.BASE`) |
|---|---|---|
| 胯 `hip` 高度 | **0.64 m** | **1.07 m** |
| 头 `hmd` 高度 | 1.353 m | 1.500 m |
| 左手 x | **+0.40**(MMD 左是正 X) | **−0.18**(我们左是负 X) |

MMD 的 `センター` 在腰上,而 AnyaDance 的 `hip` 是它自己那套标定,两者定义就不同。

✅ **所以取「相对基准帧的位移」**:`我们的姿态 = BASE + (该帧 − 基准帧) × 轴换算`。
这样**保住已经标定好的站姿**,只把舞蹈的**运动量**叠上去 ——
换模型、换动作都不用重新标定。

## 左右反了

MMD 模型**左手在 +X**,而我们的约定是**左 = 负 X**(记忆里踩过)。
所以 x 要**取负**。

## 用法

    # 只打印、不发送(看看数据对不对)
    python tools/play_dance.py <json> --dry

    # 真播(要 AnyaDance 驱动在跑,而且 **pose_server 必须停**)
    python tools/play_dance.py <json> --loop
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gestures as G  # noqa: E402

ANYA_HOST, ANYA_PORT = "127.0.0.1", 39570


def to_ours(p: list[float]) -> list[float]:
    """Blender 坐标(Z-up,米)→ 我们的坐标(Y-up),并修正左右。

    ⚠️ **x 取负** —— MMD 左手在 +X,我们左在 −X。不翻的话左右会镜像
    (举手会变成举另一只手)。
    """
    bx, by, bz = p
    return [-bx, bz, by]


def send(pose: dict[str, list[float]], sock: socket.socket,
         rot: dict[str, list[float]] | None = None) -> None:
    """发一帧。

    `rot` 是**真实旋转**(从 `dance_rot.json` 换算来的) —— 有就用真的,
    没有才退回到 `controller_rot()` 那套"从手臂方向反推"的估算。

    ⚠️ **`rotation_xyzw` 是必填的** —— 缺了驱动判 `invalid pose frame` **整帧拒收**,
    表现是"包发出去了、驱动也收到了、模型纹丝不动",只看发送端完全查不出来。
    """
    out = {"version": 1, "devices": {}}
    for name, pos in pose.items():
        if rot and name in rot:
            r = rot[name]
        elif name == "hmd":
            r = G.quat()                # 没有真实旋转时的兜底:头朝向不控
        elif name in G.BASE_ROT:
            r = G.controller_rot(name, pos)
        else:
            r = G.IDENT
        out["devices"][name] = {
            "valid": True, "connected": True,
            "pose": {"position": [float(v) for v in pos],
                     "rotation_xyzw": [float(v) for v in r]},
        }
    sock.sendto(json.dumps(out).encode("utf-8"), (ANYA_HOST, ANYA_PORT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--dry", action="store_true", help="只打印,不发")
    ap.add_argument("--loop", action="store_true", help="循环播")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.0, help="动作幅度缩放")
    ap.add_argument("--base-frame", type=int, default=0)
    ap.add_argument("--drive-hmd-rot", action="store_true",
                    help="把舞蹈的真实头部旋转也推给 HMD(⚠️ 会让视角翻滚,默认关)")
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    frames = d["frames"]
    fps = d.get("fps", 30) * args.speed
    base = frames[args.base_frame]["dev"]

    # 有没有**真实旋转**(`mmd_solve.py` 加 rot 导出之后才有)。
    # 老文件(dance.json)没有这个键 —— 那就退回 controller_rot 那套估算。
    base_rot = frames[args.base_frame].get("rot") or {}
    real_rot = bool(base_rot)

    # 预计算每一帧的**绝对设备姿态**(位置 + 真实旋转)
    seq: list[tuple[dict[str, list[float]], dict[str, list[float]]]] = []
    for fr in frames:
        dev = fr["dev"]
        pose = {}
        for name, p in dev.items():
            if name not in G.BASE:
                continue
            b = to_ours(base.get(name, p))
            c = to_ours(p)
            pose[name] = [G.BASE[name][i] + (c[i] - b[i]) * args.gain for i in range(3)]
        rot: dict[str, list[float]] = {}
        for name, qf in (fr.get("rot") or {}).items():
            qb = base_rot.get(name)
            if qb is None or name not in G.BASE:
                continue
            # 🔴 **默认不驱动 hmd 的旋转。**
            #
            # HMD 就是镜头 —— 把舞蹈里「頭」的旋转推给它,**整个视角会跟着翻滚**。
            # 症状很迷惑:画面里**名牌全是歪的**并且乱转。名牌在 VRChat 里是永远
            # 正对镜头的 billboard,**它们歪了只能是镜头在滚** —— 这个判据比
            # 盯着人物看可靠得多(踩过:一轮都以为是姿势本身的问题)。
            #
            # 想要头部跟着舞动,加 `--drive-hmd-rot`,但那样视角一定会晃。
            if name == "hmd" and not args.drive_hmd_rot:
                continue
            # 相对基准帧的变化(**骨头局部系**),再换算到我们坐标系并叠到基准朝向上。
            q_rel = G.quat_mul(G.quat_conj(qb), qf)
            rot[name] = G.mmd_rot_to_ours(q_rel, G.BASE_ROT.get(name))
        seq.append((pose, rot))

    print(f"动作 {d['source']['motion']}  模型 {d['source']['model']}")
    print(f"  {len(seq)} 帧 @ {fps:.0f}fps  → {len(seq)/fps:.1f} 秒")
    print(f"  基准帧 f{args.base_frame}  幅度 {args.gain}  速度 {args.speed}")
    print(f"  旋转: {'✅ 真实(MMD 解算)' if real_rot else '⚠️ 估算(controller_rot,老文件没带 rot)'}")

    if args.dry:
        print()
        for i in (0, len(seq) // 2, len(seq) - 1):
            p, r = seq[i]
            print(f"  f{i}: " + "  ".join(
                f"{k}=[{v[0]:+.3f},{v[1]:.3f},{v[2]:+.3f}]" for k, v in p.items()))
            if r:
                print(f"        旋转: " + "  ".join(
                    f"{k}=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}]"
                    for k, q in list(r.items())[:2]))
        return 0

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("\n开播(Ctrl+C 停)")
    try:
        while True:
            for pose, rot in seq:
                send(pose, s, rot)
                time.sleep(1.0 / fps)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n停")
    send({k: list(v) for k, v in G.BASE.items()}, s)   # 回站姿,别丢在最后一帧
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
