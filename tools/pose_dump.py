"""把 `anya_sniff.py --raw` 的录制翻译成人能看的姿势表。

## 为什么需要

回放的原始 JSON 长这样,一层套一层,肉眼看六个设备的位置基本不可能:

    {"event":"command_processed", "source":{"port":60052},
     "command":{"payload":"{\\"devices\\":{\\"hmd\\":{...}}}}"}   ← payload 还是字符串

这个工具把它摊平成一张表,**并且自动把重复的姿势合并** ——
所以「在 UI 里摆几个姿势」录一段,出来就是干净的一份对照表。

## 典型用法:记录基准站姿

`gestures.py` 的 `BASE` 是所有动作的起点,它必须**等于这套骨架的真实站姿**,
否则每个动作都会整体偏掉。而真实站姿只有驱动知道,所以要:

    # 1. 开着录制,在 AnyaDance UI 里依次摆出想要的姿势(自然站 / T-pose …)
    .venv\\Scripts\\python.exe tools/anya_sniff.py --raw > poses.txt

    # 2. Ctrl+C 停掉,然后:
    .venv\\Scripts\\python.exe tools/pose_dump.py poses.txt

⚠️ **光摆着不动是录不到的** —— 驱动会去重:新包和上一个**完全相同**就抑制。
所以必须让数值真的变(切换姿势、拖滑块),静止的姿势不会产生任何包。
"""

from __future__ import annotations

import json
import sys

DEVICES = ("hmd", "left_controller", "right_controller",
           "hip", "left_foot", "right_foot")
ROUND = 4          # 打印和去重的精度(米)。1e-4 = 0.1mm,足够区分站姿和 T-pose


def poses(path: str):
    """从录制文件里按出现顺序产出 (源端口, 设备表),重复的姿势只出一次。"""
    seen: set[tuple] = set()
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        raw = (ev.get("command") or {}).get("payload")
        if not raw:
            continue
        try:
            dev = (json.loads(raw) or {}).get("devices") or {}
        except Exception:
            continue
        if not dev:
            continue
        # 去重必须**连旋转一起算** —— 只比位置的话,两个只有手腕朝向不同的
        # 姿势会被当成同一个,而那正是手部最要命的差别(位置对、朝向错 = 手腕折)
        key = tuple(
            round(c, ROUND)
            for k in sorted(dev)
            for c in (dev[k]["pose"]["position"]
                      + dev[k]["pose"].get("rotation_xyzw", []))
        )
        if key in seen:
            continue
        seen.add(key)
        yield str((ev.get("source") or {}).get("port", "?")), dev


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    n = 0
    for n, (src, dev) in enumerate(poses(path), 1):
        print(f"\n=== 姿势 #{n}  (源端口 {src}) ===")
        for k in DEVICES:
            pose = (dev.get(k) or {}).get("pose") or {}
            pos, rot = pose.get("position"), pose.get("rotation_xyzw")
            if not pos:
                continue
            line = f"  {k:17s} [{pos[0]:+8.4f}, {pos[1]:8.4f}, {pos[2]:+8.4f}]"
            if rot:
                line += ("   rot [%+.4f %+.4f %+.4f %+.4f]" % tuple(rot))
            print(line)
    if n == 0:
        print("没解析出任何姿势。检查:①录制时姿势**有没有变化**(驱动会去重恒定值)")
        print("                          ②文件是不是 anya_sniff.py --raw 的输出")
        return 1
    print(f"\n共 {n} 种不同的姿势。")
    return 0


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
