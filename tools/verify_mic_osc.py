"""硬证「VRChat 到底有没有听到 AI 的声音」。

原理:往 CABLE Input 灌一段 TTS,同时监听 VRChat 发出来的 OSC。
VRChat 会用麦克风电平驱动 `/avatar/parameters/Voice`(0~1)和 `Viseme`,
这两个值动了 = VRChat 确实在采麦克风,而且采到的就是 AI 的声音。

    python tools/verify_mic_osc.py

前提(缺一不可):
  1. VRChat 的 OSC 开着
  2. VRChat 的 OSC **输出**配好了 —— 也就是
     %LocalLow%\\VRChat\\VRChat\\OSC\\usr_<id>\\usr_<id>.json 里得有 "outPort": 9001
     **这个文件 VRChat 只在启动时读一次,改完必须重启 VRChat。**
  3. VRChat 的麦克风选的是 `CABLE Output (VB-Audio Virtual Cable)`
     (用 `tools/osc_monitor.py` 或 VRChat 日志里的 uSpeak 行确认)

不想动 VRChat 的输出配置也行,那就改用肉眼:放 TTS 时看麦克风指示灯/模型口型。
"""

import sys
import threading
import time

sys.path.insert(0, ".")

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from src.config import load as load_cfg
from src.tts import TTS

TEXT = "测试一二三四五六七八九十,小mo正在说话,如果你能听到我,请看看嘴巴有没有动。"
WATCH = ("Voice", "Viseme")


def main() -> int:
    cfg = load_cfg()

    samples: dict[str, list[float]] = {w: [] for w in WATCH}
    stop = threading.Event()

    def on_any(address: str, *args) -> None:
        for w in WATCH:
            if w in address:
                for a in args:
                    if isinstance(a, (int, float)):
                        samples[w].append(float(a))
                print(f"    {address} = {args}", flush=True)

    disp = Dispatcher()
    disp.set_default_handler(on_any)
    try:
        server = BlockingOSCUDPServer(("127.0.0.1", 9001), disp)
    except OSError as exc:
        print(f"绑不上 9001: {exc} —— 有别的 OSC 程序占着?")
        return 1

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("监听 127.0.0.1:9001 (VRChat 的 OSC 输出)…")

    print(f"合成 TTS:{TEXT[:20]}…")
    tts = TTS(cfg.tts, cfg.audio.output_device)
    data, sr = tts.synthesize(TEXT)
    print(f"  {len(data) / sr:.2f}s @ {sr}Hz -> 输出设备 {tts.name}")

    print("开闸放音 ↓ 盯着 Voice/Viseme")
    tts.play(data, sr)
    time.sleep(1.0)
    stop.set()
    server.shutdown()

    print("\n--- 结果 ---")
    any_data = False
    for w, vals in samples.items():
        if not vals:
            print(f"  {w}: 一个值都没收到")
            continue
        any_data = True
        lo, hi = min(vals), max(vals)
        moved = "动了 ✓" if hi - lo > 1e-6 else "纹丝不动 ✗"
        print(f"  {w}: {len(vals)} 个值, 范围 {lo:.4f} ~ {hi:.4f}  -> {moved}")

    if any_data:
        print("\n>>> VRChat **确实**在采麦克风,而且采到的就是 AI 灌进 CABLE 的声音。")
    else:
        print("\n>>> 没收到任何 OSC。两种可能,按顺序排查:")
        print("    1) OSC 输出没配好 —— 检查 usr_<id>.json 里的 outPort,改完**重启 VRChat**")
        print("    2) 麦克风没选对 / VRChat 静音了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
