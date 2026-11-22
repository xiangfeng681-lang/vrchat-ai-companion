"""从 MMD 舞蹈里**切一段真实动作**,做成 `gestures.py` 能用的关键帧动作。

## 为什么要这么做

`gestures.py` 里的动作坐标全是**手编**的,而记忆里所有的手腕/手臂 bug 都是
手编坐标造成的(BASE 手高估错 45cm、手臂够不到被 IK 拉断、直线插值让手穿过肩膀)。
MMD 的 `.vmd` 是**真人动捕**,4848 帧里全是真手真脚。

## 和 `play_dance.py` 同一套换算

   我们的姿态 = BASE + (该帧 − 基准帧) × 轴换算

这样**保住已标定好的站姿**,只把动作的"运动量"叠上去 —— 换模型换动作都不用重标。

## 旋转

位置走矩阵 `M = [[-1,0,0],[0,0,1],[0,1,0]]`(即 `to_ours` 的 `[-bx, bz, by]`)。
`det(M) = +1`,说明是**纯旋转不是镜像**,所以四元数能直接共轭换算:

    q_ours = q_M ⊗ q_rel ⊗ q_M⁻¹     其中 q_M = [0, √2/2, √2/2, 0] (xyzw)

(代进旋转矩阵公式得到的正好是 M,已验算。)
`q_rel = q_基准帧⁻¹ ⊗ q_该帧` —— 相对基准帧的变化量,**和位置用完全同一个思路**。

最后叠到我们实测的 `BASE_ROT` 上:`our_rot = BASE_ROT ⊗ q_ours`(左乘还是右乘
取决于 q_rel 在哪个坐标系里表达 —— q_base⁻¹⊗q_frame 是**骨头的局部系**,所以接在右边)。

⚠️ **方向符号一律靠实测,推导全错**(记忆里栽过四次)。这里给 `--no-rot` 开关,
看着不对就关掉退回 BASE_ROT,不用改代码。

用法:
    python tools/make_gesture.py <dance.json> <起> <止> --name 挥手 --base 21.5
    python tools/make_gesture.py ... --dry            # 只看摘要
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gestures as G  # noqa: E402

# to_ours 的矩阵形式。四元数转换要用它对应的 q_M,所以两个都留着、别单独改一个。
Q_M = [0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0]      # xyzw


def to_ours(p):
    bx, by, bz = p
    return [-bx, bz, by]


def qmul(a, b):
    """四元数乘法(xyzw 顺序,和 AnyaDance 协议一致)。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def qconj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def qnorm(q):
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / n for v in q]


def qrot_vec(q, v):
    """把向量 v 用四元数 q 旋转 —— 用来做自检。"""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * (q_vec × v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="dance_rot.json(mm_solve.py 的带旋转输出)")
    ap.add_argument("start", type=float, help="起始秒")
    ap.add_argument("end", type=float, help="结束秒")
    ap.add_argument("--name", default="新动作")
    ap.add_argument("--base", type=float, default=None,
                    help="基准帧的秒数(默认=起始秒,这样第一帧就是我们的站姿)")
    ap.add_argument("--gain", type=float, default=1.0, help="幅度缩放")
    ap.add_argument("--every", type=int, default=3, help="每几帧取一个关键帧")
    ap.add_argument("--no-rot", action="store_true", help="不用真实旋转,退回 BASE_ROT")
    ap.add_argument("--out", default=None, help="写出的 JSON(默认打印)")
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    frames = d["frames"]
    fps = d.get("fps", 30)
    has_rot = "rot" in (frames[0] if frames else {})

    i0 = max(0, int(args.start * fps))
    i1 = min(len(frames) - 1, int(args.end * fps))
    ib = int((args.base if args.base is not None else args.start) * fps)
    ib = max(0, min(len(frames) - 1, ib))
    if i1 <= i0:
        print("时间范围不对")
        return 1

    base_f = frames[ib]
    base_pos = {k: to_ours(v) for k, v in base_f["dev"].items()}
    base_rot_raw = base_f.get("rot", {}) if has_rot else {}

    print(f"\n{d['source']['model']} / {d['source']['motion']}")
    print(f"切 {args.start:.1f}~{args.end:.1f}s  → 帧 {i0}~{i1} "
          f"({(i1 - i0) / fps:.2f} 秒,{fps}fps)")
    print(f"基准帧 第 {ib} 帧 ({ib / fps:.1f}s)   幅度×{args.gain}   "
          f"每 {args.every} 帧取一个关键帧")
    print(f"旋转: {'真实(MMD)' if has_rot and not args.no_rot else 'BASE_ROT(不用真实旋转)'}")

    # ---- 自检:q_M 换算和位置矩阵必须是同一套 ----
    # 随机向量的两种算法要对上:先转坐标再旋转 == 先旋转再转坐标
    v = [0.3, -0.7, 0.5]
    qb = [0.2, 0.3, -0.1, 0.9]
    qb = qnorm(qb)
    lhs = qrot_vec(qmul(qmul(Q_M, qb), qconj(Q_M)), to_ours(v))
    rhs = to_ours(qrot_vec(qb, v))
    err = max(abs(lhs[i] - rhs[i]) for i in range(3))
    print(f"\n[自检] 旋转换算与位置换算一致性误差 = {err:.2e} "
          f"{'✅' if err < 1e-6 else '❌ 公式不对'}")

    # ---- 切关键帧 ----
    steps = []
    for i in range(i0, i1 + 1, args.every):
        fr = frames[i]
        pose = {}
        for name, p in fr["dev"].items():
            if name not in G.BASE:
                continue
            c = to_ours(p)
            b = base_pos.get(name, c)
            pose[name] = [round(G.BASE[name][k] + (c[k] - b[k]) * args.gain, 4)
                          for k in range(3)]
        rot = {}
        if has_rot and not args.no_rot:
            # ⚠️ 这里**不能**写成 `if ... and name in G.BASE_ROT:` 再用 `name` ——
            #    `name` 是上面那个位置循环的**残留变量**(循环结束后等于最后一个
            #    设备名,通常是 right_foot),拿它当条件会让整个旋转块被静默跳过,
            #    产出的 rot 全是空字典。踩过:驱动收到没有 rotation_xyzw 的帧会
            #    判 `invalid pose frame` 整帧拒收,表现是"包发出去了、模型纹丝不动"。
            #    所以条件只看开关,每个设备在自己那层判断。
            for nm, qf in fr.get("rot", {}).items():
                if nm not in G.BASE:
                    continue
                qb_ = base_rot_raw.get(nm)
                if qb_ is None:
                    continue
                q_rel = qmul(qconj(qb_), qf)                 # 相对基准帧(局部系)
                q_ours = qmul(qmul(Q_M, q_rel), qconj(Q_M))  # 换到我们的坐标
                # 没有实测基准朝向的设备(头/胯/脚)用单位四元数当底 ——
                # 基准帧那一帧 q_rel 是单位阵,所以叠出来正好还是它自己,自洽。
                rot[nm] = qnorm(qmul(G.BASE_ROT.get(nm, G.IDENT), q_ours))
        steps.append((pose, rot))

    dur = args.every / fps
    print(f"关键帧 {len(steps)} 个,每个 {dur:.3f}s,合计 {len(steps) * dur:.2f}s")

    # ---- 幅度摘要:让"这个动作到底动没动"一眼可见 ----
    allp = {}
    for pose, _ in steps:
        for k, v in pose.items():
            allp.setdefault(k, []).append(v)
    print("\n各部位活动范围(转成我们的坐标之后):")
    for k in G.BASE:
        if k not in allp or len(allp[k]) < 2:
            continue
        rng = [max(v[i] for v in allp[k]) - min(v[i] for v in allp[k]) for i in range(3)]
        span = math.sqrt(sum(r * r for r in rng))
        flag = "  ← 动得多" if span > 0.3 else ""
        print(f"   {k:18} Δ=({rng[0]:.3f},{rng[1]:.3f},{rng[2]:.3f})  合位移 {span:.3f}m{flag}")

    out = {
        "name": args.name,
        "source": d["source"],
        "range": [args.start, args.end],
        "fps": round(fps / args.every, 3),
        "gain": args.gain,
        # ⚠️ 看**实际产出**的 rot,而不是看开关 —— 上一版这里写的是
        #    `has_rot and not args.no_rot`,结果旋转块被 bug 跳过、rot 全空,
        #    这个标记却照样是 True,把人往错的方向带。
        "use_rot": any(r for _, r in steps),
        "frames": [{"dur": round(dur, 4), "pose": p, "rot": r} for p, r in steps],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n→ {args.out}")
    else:
        print()
        print(json.dumps(out["frames"][0], ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
