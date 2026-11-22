"""动作库 —— 每个动作就是一串关键帧,不需要 Blender / MMD。

## 一个动作长什么样

    "挥手": [
        (0.30, pose(right_controller=WAVE_MID)),   # 把手抬起来
        (0.22, pose(right_controller=WAVE_A)),     # 摆过去
        (0.22, pose(right_controller=WAVE_B)),     # 摆回来
        ...
    ]

每项是 `(这一帧持续多少秒, 目标姿态)`。播放时在相邻两帧之间插值,
**第一帧就是起手姿势** —— 从待机切过来由 `pose_server` 做平滑过渡(见那边)。

## ⚠️ 姿态数字是实测来的,别自己编

从 AnyaDance UI 的默认站姿扒的。两个最容易搞错的点:

- **左 = 负 X,右 = 正 X**
- **脚踝在 y≈0.20,不是脚底** —— 追踪器戴在脚踝上,
  按"脚底 0.06"编会让 IK 把腿拧成麻花(实测踩过)

## 加新动作

在 `GESTURES` 里加一项就行,不用改 `pose_server`。键名就是指令名,
中文英文都行 —— 反正走的是字典查表。
"""
from __future__ import annotations

import math

# ---- 基准站姿 -----------------------------------------------------------
# **实测值** —— 2026-09-17 用 `anya_sniff.py --raw` 录下 AnyaDance UI 的
# 自然站姿,再用 `pose_dump.py` 解析出来的。不是估的,别手改。
#
# ⚠️ 改之前那版(手 y=1.28、脚 y=0.20)是**照 UI 面板估的,全部偏低/偏高**:
# 手的真实高度是 0.83,估成了 1.28 —— **高了 45cm**。
# 所有动作的第 0 帧都从这个 BASE 插值过去,所以手会先"啪"地掉 45cm 再做动作。
# 这类错误看代码发现不了,只能实测。
#
# 怎么重新测(换 Avatar / 重新校准 FBT 之后要做):
#   1. 开着录制:tools/anya_sniff.py --raw > poses.txt
#   2. 在 AnyaDance UI 里切一下姿势(⚠️ 静止不动录不到,驱动会去重)
#   3. tools/pose_dump.py poses.txt
# 🔄 2026-09-18 **重测**:只有手变了,其余(头 1.500 / 胯 1.070 / 脚 0.260)一模一样。
#    旧值手是 [±0.180, 0.830],新测 [±0.272, 0.628] —— **低了 20.2cm、外了 9.2cm**。
#    测法:`tools/anya_sniff.py --raw > poses.txt` 边录边在 UI 里切姿势,
#    再 `tools/pose_dump.py poses.txt`;原始录制存在
#    `_mmd/poses_20260918.txt` 与 `_mmd/poses_tpose_20260918.txt`。
#
#    ⚠️ **T-pose 两次测都一样**([±0.680, 1.330]),所以**骨架没变**,变的只是站姿。
#    但要注意:用新的站姿 + T-pose 联立反推肩膀,会得到 sy=1.151 ——
#    只比胯(1.070)高 8cm,**解剖上不可能**。说明这个"站姿"和旧的那组
#    不是同一套模型下的姿势(可能手臂角度不同)。**SHOULDER/ARM_LEN 因此没动**,
#    它们被 T-pose 单独验证过(两个姿势都给出 L=0.505)。
BASE = {
    "hmd": [0.0, 1.500, 0.0],
    "left_controller": [-0.272, 0.628, -0.10],
    "right_controller": [0.272, 0.628, -0.10],
    "hip": [0.0, 1.070, -0.05],
    "left_foot": [-0.09, 0.260, 0.10],
    "right_foot": [0.09, 0.260, 0.10],
}

# ---- 各设备的基准朝向(四元数 XYZW)---------------------------------------
# 同样实测。**手柄那两条不是单位四元数**,这是踩出来的:
#
# 第一版对所有设备一律发 `[0,0,0,1]`(不旋转),因为 `hip` 和双脚实测
# 确实就是单位四元数,就以为全体如此。**手柄不是** ——
# 站姿下真实值是绕轴约 27° 的旋转(手掌自然朝向),T-pose 下是绕 Z ±90°。
#
# VRChat 的手部 IK 是**位置 + 朝向一起吃**的:位置说"手举到这儿"、
# 朝向说"手腕别转",两个信息打架 → 手腕被拧断。
#
# ⚠️ 左右手不是简单的 x 取反:**x 和 w 相同,y 和 z 反号**。
BASE_ROT = {
    "left_controller": [-0.2105, 0.0902, -0.0501, 0.9721],
    "right_controller": [-0.2105, -0.0902, 0.0501, 0.9721],
}


def rot_of(device: str) -> list[float]:
    """设备的基准朝向。hmd 不在这里 —— 它的朝向是每帧按偏航/俯仰算的。"""
    return list(BASE_ROT.get(device, IDENT))


# ---- 四元数小工具(自带,不引 numpy)--------------------------------------

def _norm3(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else [0.0, 0.0, 1.0]


def _dot3(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross3(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def quat_from_to(a, b) -> list[float]:
    """把向量 a 转到 b 的**最短**旋转(四元数 XYZW)。"""
    a, b = _norm3(a), _norm3(b)
    d = _dot3(a, b)
    if d > 1 - 1e-9:
        return list(IDENT)
    if d < -1 + 1e-9:
        # 正好反向:最短路径不唯一,挑一个垂直于 a 的轴转 180°
        axis = _cross3(a, [1.0, 0.0, 0.0])
        if math.sqrt(sum(x * x for x in axis)) < 1e-6:
            axis = _cross3(a, [0.0, 1.0, 0.0])
        axis = _norm3(axis)
        return [axis[0], axis[1], axis[2], 0.0]
    axis = _cross3(a, b)
    return _norm3([axis[0], axis[1], axis[2], 1.0 + d])


def quat_mul(p, q) -> list[float]:
    """四元数乘法 `p * q` —— 表示**先施加 q,再施加 p**。顺序不能反。"""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return [
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
        pw * qw - px * qx - py * qy - pz * qz,
    ]


# ---- MMD 解算结果的坐标换算 ------------------------------------------------
# `to_ours` 的 `(bx,by,bz) → (-bx, bz, by)` 对应矩阵 **det = +1**,
# 是**纯旋转不是镜像**,所以四元数能用共轭换算,不用改公式:
#
#     q_ours = Q_M ⊗ q ⊗ Q_M⁻¹        Q_M = [0, √2/2, √2/2, 0] (xyzw)
#
# (把 Q_M 代进旋转矩阵公式,得到的正好是那个矩阵 —— 已验算。)
#
# 自检手法(可复用):随机向量「先转坐标再旋转」必须等于「先旋转再转坐标」,
# 实测误差 2e-16。⚠️ 但它**只验公式,验不了方向符号** —— 符号只能肉眼看。
Q_M = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]


def quat_conj(q) -> list[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def quat_norm(q) -> list[float]:
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / n for v in q]


def mmd_rot_to_ours(q_rel, base_rot=None) -> list[float]:
    """MMD 骨骼的**相对基准帧旋转** → 我们坐标系下的绝对朝向。

    `q_rel = q_基准帧⁻¹ ⊗ q_该帧` —— 是在**骨头的局部系**里表达的,
    所以结果接在 `base_rot` 的**右边**(先转到基准朝向,再叠局部变化)。

    没给 `base_rot` 就用单位阵 —— 头/胯/脚没有实测基准朝向,本来就是 IDENT。
    """
    q = quat_mul(quat_mul(Q_M, q_rel), quat_conj(Q_M))
    return quat_norm(quat_mul(list(base_rot) if base_rot else list(IDENT), q))


def controller_rot(device: str, hand_pos) -> list[float]:
    """手柄朝向 = **基准朝向跟着手臂一起转**。

    为什么不能像 `hip`/脚那样用一个固定值:

    `BASE_ROT` 那组数字对应的是**手臂垂下**时的手腕角度。手臂一举起来,
    手的位置从 y=0.83 跑到 y=1.73,而朝向还冻在垂下时的角度 ——
    手腕**相对手臂**就被扭过了一大截,看起来就是"腕子折了"。

    这里把「手臂从基准方向转到当前方向」的那段旋转叠加上去,于是手腕
    **相对手臂**的角度恒定不变,手臂转到哪手腕就跟到哪 —— 这才是自然的。
    """
    base_rot = BASE_ROT.get(device)
    if not base_rot:
        return list(IDENT)
    sh = SHOULDER[device]
    d0 = [BASE[device][i] - sh[i] for i in range(3)]      # 基准:手臂垂下
    d1 = [hand_pos[i] - sh[i] for i in range(3)]          # 当前:手指哪
    return quat_mul(quat_from_to(d0, d1), base_rot)


def pose(**over) -> dict:
    """从基准站姿派生一个姿态。`over` 直接写设备名。

        pose(hip=[0, 0.95, -0.05], right_controller=[0.45, 1.85, -0.25])
    """
    d = {k: list(v) for k, v in BASE.items()}
    for k, v in over.items():
        if k in d:
            d[k] = [float(x) for x in v]
    return d


def _slerp_dir(d0, d1, u):
    """两个**单位**向量之间的球面插值。"""
    dot = max(-1.0, min(1.0, _dot3(d0, d1)))
    theta = math.acos(dot)
    if theta < 1e-6:
        return list(d0)
    s = math.sin(theta)
    return [(math.sin((1 - u) * theta) / s) * d0[i]
            + (math.sin(u * theta) / s) * d1[i] for i in range(3)]


def _lerp_hand(device: str, p0, p1, u) -> list[float]:
    """手柄位置沿**绕肩膀的球面**插值,而不是直线。

    为什么不能直线插值(实测踩到的):

    挥手收尾要把手从 `y=1.727` 放回 `y=0.83`,直线插值的**中点落在离肩膀
    只有 0.18m 的地方**(两端分别是 0.49 和 0.505)—— 手臂被强行折起来再展开。
    更要命的是手经过肩膀附近时 `hand - shoulder` 接近零向量,**方向计算会
    剧烈翻转**:实测相邻帧手部朝向跳 **30°**,在 60Hz 下就是手腕在抽。

    绕肩膀转 + 半径单独插值,手永远不会穿过肩膀,朝向也就始终平滑。
    """
    sh = SHOULDER[device]
    v0 = [p0[i] - sh[i] for i in range(3)]
    v1 = [p1[i] - sh[i] for i in range(3)]
    r0 = math.sqrt(sum(x * x for x in v0))
    r1 = math.sqrt(sum(x * x for x in v1))
    if r0 < 1e-6 or r1 < 1e-6:
        return [p0[i] + (p1[i] - p0[i]) * u for i in range(3)]
    r = r0 + (r1 - r0) * u
    d = _slerp_dir([x / r0 for x in v0], [x / r1 for x in v1], u)
    return [sh[i] + d[i] * r for i in range(3)]


def lerp_pose(a: dict, b: dict, u: float) -> dict:
    """两个姿态之间插值(u 从 0 到 1)。

    手柄走球面(见 `_lerp_hand`),其余设备走直线 —— 头/胯/脚的运动本来
    就近似平移,不存在"绕某个关节转"的问题。
    """
    out = {}
    for k in BASE:
        if k in SHOULDER:
            out[k] = _lerp_hand(k, a[k], b[k], u)
        else:
            out[k] = [a[k][i] + (b[k][i] - a[k][i]) * u for i in range(3)]
    return out


def smoothstep(x: float) -> float:
    """两端速度为零的平滑曲线 —— 用它插值才没有机械的匀速感。"""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def quat(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> list[float]:
    """偏航/俯仰/横滚(度) → 四元数 XYZW。只给头部用。

    🐛 **这里曾经把 x 和 y 两行写反了**(2026-09-17 修)。
    后果:**`quat(yaw=θ)` 实际做的是"点头"(绕 X 轴),不是"转头"**。
    当时没被发现,是因为唯一的老调用点只转 ±6°(待机时头部微动)——
    点头 6° 和转头 6° 肉眼几乎分不出。

    做"转向说话的人"(要转几十度)时才暴露:头往一边点、身体往一边转,
    拧成麻花。**判据**:纯偏航不该改变旋转后向量的 y 分量 ——
    拿 (0,0,1) 转一下,看它落在 XY 还是 XZ 平面上,一试就知道。

    正确的 `q = qy ⊗ qx ⊗ qz` 是下面这样(x 收俯仰项、y 收偏航项)。
    """
    cy, sy = math.cos(math.radians(yaw) / 2), math.sin(math.radians(yaw) / 2)
    cp, sp = math.cos(math.radians(pitch) / 2), math.sin(math.radians(pitch) / 2)
    cr, sr = math.cos(math.radians(roll) / 2), math.sin(math.radians(roll) / 2)
    # q = qy * qx * qz
    x = cy * sp * cr + sy * cp * sr          # 俯仰 → x
    y = sy * cp * cr - cy * sp * sr          # 偏航 → y  ← 这两行以前是反的
    z = cy * cp * sr - sy * sp * cr
    w = cy * cp * cr + sy * sp * sr
    return [x, y, z, w]


IDENT = [0.0, 0.0, 0.0, 1.0]


def rotate_y(pose: dict, yaw_deg: float,
             *, keep: tuple = ("left_foot", "right_foot")) -> dict:
    """把整个姿势绕**垂直轴**转 `yaw_deg` 度。用于「转向说话的人」。

    🔑 **绕原点(头正下方)转就够了。** 各点离这条轴本来就都很近 ——
    胯 0.05m、头 0m、手 0.21m。所以转 30°,手也只挪 10cm,
    看起来就是"原地转身",没有明显的滑动感。

    ⚠️ **双脚默认不转。** 真人转身是上半身先转、脚再挪步;而我们没有脚部
    追踪器,硬转双脚会像"整个人在冰上打滑"。脚钉住、只转上半身,
    在小角度下反而更像真人。

    ⚠️ 只做**位置**旋转;朝向(四元数)由调用方另算 —— 头部的偏航是
    直接加在 `quat(yaw=...)` 上的,手柄朝向则由 `controller_rot()` 按
    新的手臂方向重新推。
    """
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    out = {k: list(v) for k, v in pose.items()}
    for k, v in out.items():
        if k in keep:
            continue
        x, y, z = v
        out[k] = [x * c + z * s, y, -x * s + z * c]
    return out

# ---- 手臂可达范围 ---------------------------------------------------------
# 肩膀位置和臂长是**联立解出来的**,不是量的(手上没有真 VR 设备)。
#
# 用实测的两个姿势,假设两者手臂都伸直(长度 L 相同):
#     T-pose: (0.68-sx)² + (1.33-sy)² + (-0.10-sz)² = L²
#     站姿:   (0.18-sx)² + (0.83-sy)² + (-0.10-sz)² = L²
#   相减 → **sx + sy = 1.51**
#
# 一条线有无穷多解,需要第二个约束。肩膀必定在头部下方:hmd 在 y=1.5,
# 正常肩高约低 17cm → 取 sy = 1.33,于是 sx = 0.18。
# 代回去两个姿势**都得到 L = 0.505** —— 自洽,所以这组值可信。
SHOULDER = {
    "left_controller": (-0.18, 1.33, -0.03),
    "right_controller": (0.18, 1.33, -0.03),
}
ARM_LEN = 0.505

# ⚠️ 别设成 1.0 —— 贴死极限会让解算器在"伸得到/伸不到"之间抖。
# 留 3% 是因为**人举手时手臂本来就不必完全伸直**,微弯更自然。
REACH_MARGIN = 0.97


def reach(target: list[float]) -> list[float]:
    """把够不到的手部目标**沿原方向拉回**到臂长范围内。

    为什么必须做:VRChat 的手臂 IK 接到超过臂长的目标时**不报错也不拒绝**,
    它会把手肘拉直、手腕扭转去硬凑 —— 看起来就是"手折了"。

    实测的翻车现场:`_HAND_R_UP` 原本要 0.59m、`_WAVE_A` 要 **0.70m**,
    而臂长只有 0.50m。挥手超标 **40%**,于是手被拧断。

    **沿原方向缩放、而不是截断** —— 保住设计意图(手往哪个方向抬),
    只改"够多远"。左右由 x 的符号自动判断。
    """
    side = "left_controller" if target[0] < 0 else "right_controller"
    sh = SHOULDER[side]
    v = [target[i] - sh[i] for i in range(3)]
    d = math.sqrt(sum(x * x for x in v))
    limit = ARM_LEN * REACH_MARGIN
    if d <= limit or d == 0:
        return [float(x) for x in target]
    k = limit / d
    return [sh[i] + v[i] * k for i in range(3)]


# ---- 常用手势点位(左负右正,高度单位米)-------------------------------------
# ⚠️ 一律经 `reach()` 兜底 —— 手写绝对坐标极易超出臂长,而超了就是"手折"
_HAND_R_UP = reach([0.42, 1.80, -0.30])    # 右手举起
_HAND_L_UP = reach([-0.42, 1.80, -0.30])   # 左手举起
_WAVE_A = reach([0.58, 1.86, -0.26])       # 挥手:摆到外侧
_WAVE_B = reach([0.28, 1.78, -0.34])       # 挥手:摆回内侧

# 一个动作 = [(持续秒数, 目标姿态, 头部偏航°, 头部俯仰°), ...]
GESTURES: dict[str, list[tuple]] = {
    # ---- 单手 ----------------------------------------------------------
    "举手": [
        (0.32, pose(right_controller=_HAND_R_UP), 4, -4),
        (1.30, pose(right_controller=_HAND_R_UP), 4, -4),
        (0.36, pose(), 0, 0),
    ],
    "挥手": [
        (0.30, pose(right_controller=_HAND_R_UP), 5, -3),
        (0.22, pose(right_controller=_WAVE_A), 7, -3),
        (0.22, pose(right_controller=_WAVE_B), 4, -3),
        (0.22, pose(right_controller=_WAVE_A), 7, -3),
        (0.22, pose(right_controller=_WAVE_B), 4, -3),
        (0.34, pose(), 0, 0),
    ],
    # ---- 头部 ----------------------------------------------------------
    "点头": [
        (0.24, pose(hmd=[0.0, 1.470, 0.02]), 0, 14),
        (0.24, pose(hmd=[0.0, 1.505, 0.0]), 0, -4),
        (0.24, pose(hmd=[0.0, 1.470, 0.02]), 0, 14),
        (0.24, pose(hmd=[0.0, 1.505, 0.0]), 0, -2),
        (0.20, pose(), 0, 0),
    ],
    "摇头": [
        (0.24, pose(), -18, 0),
        (0.24, pose(), 18, 0),
        (0.24, pose(), -18, 0),
        (0.24, pose(), 18, 0),
        (0.20, pose(), 0, 0),
    ],
    # ---- 双手 ----------------------------------------------------------
    "拍手": [
        (0.26, pose(left_controller=[-0.10, 1.42, -0.42],
                    right_controller=[0.10, 1.42, -0.42]), 0, 2),
        (0.36, pose(left_controller=[-0.14, 1.40, -0.36],
                    right_controller=[0.14, 1.40, -0.36]), 0, 2),
        (0.26, pose(left_controller=[-0.10, 1.42, -0.42],
                    right_controller=[0.10, 1.42, -0.42]), 0, 2),
        (0.36, pose(left_controller=[-0.14, 1.40, -0.36],
                    right_controller=[0.14, 1.40, -0.36]), 0, 2),
        (0.26, pose(left_controller=[-0.10, 1.42, -0.42],
                    right_controller=[0.10, 1.42, -0.42]), 0, 2),
        (0.30, pose(), 0, 0),
    ],
    "摊手": [
        (0.34, pose(left_controller=[-0.40, 1.30, -0.44],
                    right_controller=[0.40, 1.30, -0.44]), 0, -2),
        (1.00, pose(left_controller=[-0.40, 1.30, -0.44],
                    right_controller=[0.40, 1.30, -0.44]), 0, -2),
        (0.34, pose(), 0, 0),
    ],
    "叉腰": [
        (0.36, pose(left_controller=[-0.30, 1.10, -0.16],
                    right_controller=[0.30, 1.10, -0.16]), 0, 0),
        (1.40, pose(left_controller=[-0.30, 1.10, -0.16],
                    right_controller=[0.30, 1.10, -0.16]), 0, 0),
        (0.36, pose(), 0, 0),
    ],
    "思考": [
        (0.40, pose(right_controller=[0.14, 1.52, -0.26],
                    hmd=[0.02, 1.495, 0.0]), -6, 6),
        (1.30, pose(right_controller=[0.14, 1.52, -0.26],
                    hmd=[0.02, 1.495, 0.0]), -6, 6),
        (0.36, pose(), 0, 0),
    ],
    "鞠躬": [
        (0.45, pose(hmd=[0.0, 1.34, 0.16], hip=[0.0, 0.94, -0.02]), 0, 26),
        (0.90, pose(hmd=[0.0, 1.34, 0.16], hip=[0.0, 0.94, -0.02]), 0, 26),
        (0.45, pose(), 0, 0),
    ],
    "双手举手": [
        (0.34, pose(left_controller=_HAND_L_UP, right_controller=_HAND_R_UP), 0, -6),
        (1.20, pose(left_controller=_HAND_L_UP, right_controller=_HAND_R_UP), 0, -6),
        (0.36, pose(), 0, 0),
    ],
}

# 别名 —— 让指令词宽松一点,LLM 吐哪个都能对上
ALIASES = {
    "打招呼": "挥手", "拜拜": "挥手", "再见": "挥手",
    "你好": "挥手", "hi": "挥手", "hello": "挥手",
    "鼓掌": "拍手", "啪啪": "拍手",
    "耸肩": "摊手", "不知道": "摊手", "无奈": "摊手",
    "是的": "点头", "同意": "点头", "嗯": "点头",
    "不是": "摇头", "拒绝": "摇头", "不": "摇头",
    "托腮": "思考", "想想": "思考",
    "弯腰": "鞠躬", "谢谢": "鞠躬",
    "万岁": "双手举手", "欢呼": "双手举手", "举手手": "举手",
}


def resolve(name: str) -> str | None:
    """把指令词翻译成动作名。对不上返回 None(调用方当作"没有这个动作")。"""
    name = (name or "").strip()
    if name in GESTURES:
        return name
    return ALIASES.get(name)


def gesture_at(name: str, t: float) -> tuple[dict, list[float]] | None:
    """动作 `name` 在自身时间 t 秒时的 (姿态, 头部四元数)。

    末帧之后返回 None(动作已结束)。
    """
    keys = GESTURES.get(name)
    if not keys:
        return None
    acc = 0.0
    for i, (dur, p, yaw, pitch) in enumerate(keys):
        if t < acc + dur or i == len(keys) - 1:
            if i == len(keys) - 1 and t >= acc + dur:
                return None                      # 整段放完了
            nxt = keys[min(i + 1, len(keys) - 1)]
            u = smoothstep((t - acc) / dur) if dur > 0 else 1.0
            p2, y2, pi2 = nxt[1], nxt[2], nxt[3]
            cur = lerp_pose(p, p2, u)
            yaw_c = yaw + (y2 - yaw) * u
            pitch_c = pitch + (pi2 - pitch) * u
            return cur, quat(yaw=yaw_c, pitch=pitch_c)
        acc += dur
    return None


def duration(name: str) -> float:
    """动作总时长(秒)。"""
    return sum(k[0] for k in GESTURES.get(name, []))
