"""音频设备枚举与解析。

VRChat AI 需要两个设备,别搞混:

  耳朵 capture_device —— 听到玩家说话。两种来源:
     方案 A  VB-Audio Cable:VRChat 输出 → CABLE Input,录 CABLE Output
             (填 "CABLE Output"。缺点:自己听不到游戏声音)
     方案 B  WASAPI 回环:直接抓扬声器,设备名带 " [Loopback]"
             (填 "loopback" 或扬声器名片段。优点:自己正常听得到)

  嗓子 output_device  —— AI 说话。填 VB-Audio 的 "CABLE Input"
             (注意是 Input 那端,因为对我们来说是「播放到」它)
"""

from __future__ import annotations

import sounddevice as sd

# 同一个物理设备会在 MME / DirectSound / WASAPI / WDM-KS 下各出现一次。
# 优先选 WASAPI —— 延迟最低,也是 Windows 现在的标准接口。
# (VB-CABLE 这种虚拟线尤其明显,一个 "CABLE Input" 会匹配到 3 个。)
_HOSTAPI_PREFERENCE = (
    "Windows WASAPI",
    "Windows DirectSound",
    "MME",
    "Windows WDM-KS",
)


def _hostapi_rank(dev: dict) -> int:
    try:
        return _HOSTAPI_PREFERENCE.index(dev["hostapi"])
    except ValueError:
        return len(_HOSTAPI_PREFERENCE)


def query() -> list[dict]:
    """把所有设备整理成好读的列表。"""
    hostapis = sd.query_hostapis()
    out: list[dict] = []
    for idx, dev in enumerate(sd.query_devices()):
        try:
            api = hostapis[dev["hostapi"]]["name"]
        except Exception:
            api = "?"
        out.append({
            "index": idx,
            "name": dev["name"],
            "hostapi": api,
            "in": dev["max_input_channels"],
            "out": dev["max_output_channels"],
            "rate": int(dev["default_samplerate"]),
            "is_loopback": "loopback" in dev["name"].lower(),
        })
    return out


def default(kind: str) -> int | None:
    """kind: 'input' | 'output'。返回系统默认设备索引。"""
    try:
        return sd.default.device[0 if kind == "input" else 1]
    except Exception:
        return None


def resolve(spec, kind: str) -> int | None:
    """把配置里的设备写法解析成索引。

    spec 可以是:None(用默认) / int(索引) / str(名字片段,不区分大小写)。
    kind: 'input' | 'output'
    """
    ch_key = "in" if kind == "input" else "out"

    if spec is None or spec == "":
        return default(kind)

    devices = query()

    if isinstance(spec, int):
        return spec

    needle = str(spec).strip().lower()

    # 特殊关键词:自动找回环设备
    if needle in ("loopback", "回环"):
        if kind != "input":
            raise ValueError("回环只能当输入(capture_device)用")
        cands = [d for d in devices if d["is_loopback"] and d[ch_key] > 0]
        if not cands:
            raise ValueError(
                "没找到 WASAPI 回环设备。请确认:\n"
                "  1) Windows 上跑(回环是 WASAPI 特性)\n"
                "  2) 系统里有可用的扬声器输出\n"
                "  3) 改用方案 A:装 VB-Audio Cable,填 \"CABLE Output\""
            )
        return cands[0]["index"]

    matches = [
        d for d in devices
        if needle in d["name"].lower() and d[ch_key] > 0
    ]
    if not matches:
        raise ValueError(
            f"没找到名字含 {spec!r} 的{kind}设备。"
            f"跑 `main.py devices` 看完整列表。"
        )

    # 同名设备在多个 host API 下会重复出现 —— 按偏好排序,取最好的那个 API,
    # 只在**同一个** API 下还重名时才报错。
    matches.sort(key=_hostapi_rank)
    best_api = matches[0]["hostapi"]
    same_api = [m for m in matches if m["hostapi"] == best_api]

    if len(same_api) > 1:
        names = [f"  [{m['index']}] {m['name']} ({m['hostapi']})" for m in same_api]
        raise ValueError(
            f"{spec!r} 在 {best_api} 下匹配到多个设备,请写得更具体或用索引号:\n"
            + "\n".join(names)
        )
    return same_api[0]["index"]


def describe(index: int | None) -> str:
    if index is None:
        return "(未设置)"
    try:
        d = sd.query_devices(index)
        return f"[{index}] {d['name']}"
    except Exception:
        return f"[{index}] (无效索引)"
