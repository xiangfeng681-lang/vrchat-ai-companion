"""用 Blender + MMD Tools **无头解算** MMD 动作,导出成我们 6 个追踪点的逐帧姿态。

## 为什么要自己解算,而不是用 AnyaDance 的 Dance (MMD) 对话框

AnyaDance 那个对话框能用,但**只能在 GUI 里点**,结果存成 `.nya` 之后就固定了。
自己解算的好处:

1. **程序可控** —— 可以接进语音(说一句"跳个舞"就跳)、可以循环、可以调速
2. **能直接喂给 `pose_server`** —— 它本来就在推同样这六个设备,不用多一条链路
3. **能自己标定坐标** —— 换模型/换动作时不用重新在 GUI 里摸

## MMD 骨骼 → 我们的六个设备

| 设备 | MMD 骨头 | 说明 |
|---|---|---|
| `hmd` | **頭** | 头显 = 脑袋 |
| `left_controller` | **手首.L** | ⚠️ 不是「左手首」,MMD Tools 用 `.L`/`.R` **后缀** |
| `right_controller` | **手首.R** | |
| `hip` | **センター** | 中心骨(胯) |
| `left_foot` | **足首.L** | 脚踝 |
| `right_foot` | **足首.R** | |

## 坐标转换

MMD Tools 按 `scale=0.08` 导入后是 **Blender 的 Z-up / 米**,而我们是
**Y-up**(VRChat/Unity 那套)。所以 (bx, by, bz) → (bx, bz, by) 还是取负,
**得实测** —— 先跑一遍看头像朝哪边。

## 用法

    blender.exe --background --python tools/mmd_solve.py -- <模型.pmx> <动作.vmd> <输出.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # type: ignore

# MMD 骨头名 → 我们的设备名
BONE_MAP = {
    "hmd": "頭",
    "left_controller": "手首.L",
    "right_controller": "手首.R",
    "hip": "センター",
    "left_foot": "足首.L",
    "right_foot": "足首.R",
}


def _args() -> list[str]:
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def main() -> int:
    a = _args()
    if len(a) < 3:
        print("用法: -- <模型.pmx> <动作.vmd> <输出.json>")
        return 1
    pmx, vmd, out = a[0], a[1], a[2]

    # ⚠️ **不能用 `wm.read_factory_settings`** —— 它会重置偏好设置,
    #    连 MMD Tools 扩展一起禁掉,后面 `bpy.ops.mmd_tools.*` 会报
    #    "could not be found"。手动删对象就行。
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)

    print(f"[solve] 导入模型 {Path(pmx).name} …")
    bpy.ops.mmd_tools.import_model(filepath=pmx, scale=0.08,
                                   types={"MESH", "ARMATURE", "MORPHS"})
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        print("[solve] 没找到骨架,失败")
        return 1
    have = {b.name for b in arm.data.bones}
    missing = [b for b in BONE_MAP.values() if b not in have]
    if missing:
        print(f"[solve] ⚠️ 模型缺这些骨头: {missing}")
    print(f"[solve] 骨架 {arm.name},{len(have)} 根骨头")

    print(f"[solve] 导入动作 {Path(vmd).name} …")
    # ⚠️ 参数名以 `get_rna_type()` 为准 —— 文档/记忆里的 `clean_animation` 不存在。
    # `update_scene_settings=True` 才会**按动作的长度自动设帧范围**,
    # 不设的话场景还停在默认的 1~250 帧。
    bpy.ops.mmd_tools.import_vmd(filepath=vmd, scale=0.08,
                                 update_scene_settings=True)

    sc = bpy.context.scene
    f0, f1 = int(sc.frame_start), int(sc.frame_end)
    print(f"[solve] 帧范围 {f0} ~ {f1}")

    frames = []
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        bpy.context.view_layer.update()
        dev = {}
        rot = {}
        for dev_name, bone in BONE_MAP.items():
            pb = arm.pose.bones.get(bone)
            if pb is None:
                continue
            # head = 骨头起点(关节位置),世界坐标 = 骨架矩阵 @ 姿态空间点
            # 注意:姿态变换要算进去 —— pose_bone.matrix 已经是姿态后的
            # 骨架空间矩阵,再乘 arm.matrix_world 就是世界坐标。
            mw = arm.matrix_world @ pb.matrix
            w = mw @ __import__("mathutils").Vector((0, 0, 0))
            dev[dev_name] = [round(float(w.x), 5), round(float(w.y), 5), round(float(w.z), 5)]
            # 旋转:整个世界矩阵的四元数。⚠️ Blender 的 Quaternion 是 **(w,x,y,z)**,
            # 而 AnyaDance 协议要 **xyzw** —— 这里就按协议的顺序存,别到用的时候再忘。
            q = mw.to_quaternion()
            rot[dev_name] = [round(float(q.x), 6), round(float(q.y), 6),
                             round(float(q.z), 6), round(float(q.w), 6)]
        # `dev` 保持原样(还是位置列表)—— play_dance.py 等消费者靠它,别改结构。
        # 旋转另开一个平行的 `rot`,老代码不认这个键、不受影响。
        frames.append({"f": f - f0, "dev": dev, "rot": rot})

    meta = {
        "source": {"model": Path(pmx).name, "motion": Path(vmd).name},
        "fps": int(sc.render.fps),
        "bone_map": BONE_MAP,
        "axis": "blender 的 Z-up,米。转成 Y-up 见 _to_yup()",
        "frames": frames,
    }
    Path(out).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[solve] 导出 {len(frames)} 帧 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
