"""VRChat AI 虚拟伙伴 —— 命令行入口。

分阶段验证,别一上来就 run:
    python main.py devices          # 1. 看设备清单,填 config.yaml
    python main.py check            # 2. 检查配置有没有明显问题
    python main.py osc 你好          # 3. 验证 OSC 通路(去 VRChat 看头顶有没有字幕)
    python main.py ear              # 4. 验证耳朵(放点声音,看电平条动不动)
    python main.py tts 你好          # 5. 验证嗓子(听不听得到)
    python main.py chat             # 6. 验证大脑(纯文字,不碰音频)
    python main.py asr              # 7. 验证转写(说一句,看转写对不对)
    python main.py run              # 8. 全流程
"""

from __future__ import annotations

import argparse
import sys
import time

# Windows 中文控制台:重定向时默认会用 GBK 编码,中文会炸。
# stdin 也要 —— 管道喂中文时它按 GBK 解,末字节落单成 \udc80(surrogateescape),
# 再发往 API 就是 UnicodeEncodeError。真人在控制台敲键盘不走这条路,不受影响。
for _s in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import load, validate  # noqa: E402


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------

def cmd_devices(cfg) -> int:
    from src.audio import devices

    rows = devices.query()
    print(f"\n{'索引':>4}  {'类型':<6} {'采样率':>6}  名称 (sounddevice / PortAudio)")
    print("-" * 78)
    for d in rows:
        kinds = []
        if d["in"]:
            kinds.append("输入")
        if d["out"]:
            kinds.append("输出")
        print(f"{d['index']:>4}  {'/'.join(kinds):<6} {d['rate']:>6}  {d['name']} ({d['hostapi']})")

    print("\n=== WASAPI 回环设备(pyaudiowpatch)===")
    try:
        import pyaudiowpatch as pa

        p = pa.PyAudio()
        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        print(f"当前系统默认输出: {default_out['name']}\n")
        for lb in p.get_loopback_device_info_generator():
            mark = " ★默认" if default_out["name"] in lb["name"] else ""
            print(f"{lb['index']:>4}  {'回环':<6} {int(lb['defaultSampleRate']):>6}  "
                  f"{lb['name']}{mark}")
        p.terminate()
    except Exception as exc:
        print(f"  (取回环设备失败: {type(exc).__name__}: {exc})")

    print("\n填进 config.yaml:")
    print('  audio.capture_device ← "loopback" 就抓系统默认输出;多声卡时写 "loopback:名字片段"')
    print('  audio.output_device  ← 装完 VB-CABLE 后填 "CABLE Input"')
    print("\n⚠️ 回环和 TTS 走的必须是两条不同的路,否则 AI 会听见自己,无限自激。")
    print("   回环抓扬声器、TTS 走 CABLE,天然分得开。")
    return 0


def cmd_check(cfg) -> int:
    print(f"\n配置文件: {cfg.config_path}")
    print(f"大脑: {cfg.llm.model} @ {cfg.llm.base_url}")
    print(f"耳朵: {cfg.audio.capture_device!r}")
    print(f"嗓子: {cfg.audio.output_device!r}")

    warnings = validate(cfg)
    if not warnings:
        print("\n✅ 没发现问题。")
        return 0

    print()
    for w in warnings:
        print(f"⚠️   {w}")
    return 0


def cmd_osc(cfg, text: str) -> int:
    from src.osc_client import VRChatOSC

    osc = VRChatOSC(cfg)
    text = text or "OSC 通路测试 ✓"
    print(f"发送 /chatbox/input -> {cfg.osc.host}:{cfg.osc.port}")
    print(f"内容: {text!r}")
    ok = osc.chatbox(text, force=True)
    if not ok:
        print("(没发出去 —— 检查 chatbox.enabled 是不是 false)")
        return 1
    print("\n去 VRChat 看头顶聊天框。没出现的话依次检查:")
    print("  1) VRChat 里 Action Menu → Options → OSC → Enabled")
    print("  2) 端口是不是 9000(config.yaml 的 osc.port)")
    print("  3) 防火墙有没有拦 UDP")
    return 0


def cmd_ear(cfg) -> int:
    """实时电平表 —— 验证「耳朵」到底有没有听到声音。"""
    from src.audio.capture import Capture
    from src.audio.vad import rms

    cap = Capture(cfg.audio.capture_device, cfg.audio.block_ms, cfg.audio.preferred_rate)
    threshold = cfg.vad.energy_threshold
    print(f"耳朵: {cap.name}")
    print(f"采样率: {cap.sample_rate}Hz   块大小: {cap.block_size} 样本")
    print(f"VAD 门限: {threshold}  (低于它的块会被当成静音)")
    print("\n现在放点声音 —— VRChat 里说句话、放首歌、随便什么都行。")
    print("看下面的电平条动不动。Ctrl+C 退出。\n")

    scale = max(threshold * 4, 1e-6)
    with cap:
        try:
            while True:
                block = cap.queue.get()
                level = rms(block)
                bars = int(min(1.0, level / scale) * 40)
                flag = "  ◀ 超过门限" if level >= threshold else ""
                print(f"\r[{'█' * bars}{'·' * (40 - bars)}] {level:.5f}{flag}     ",
                      end="", flush=True)
        except KeyboardInterrupt:
            print(f"\n\n退出。丢帧 {cap.dropped} 次。")
    return 0


def cmd_tts(cfg, text: str) -> int:
    from src.tts import TTS

    tts = TTS(cfg.tts, cfg.audio.output_device)
    print(f"输出设备: {tts.name}")
    # ⚠️ cfg.tts.voice 只在 backend=edge 时才有意义(那是微软的音色名)。
    # 另外两个后端音色由别的东西决定,直接打 voice 会误导。
    if cfg.tts.backend == "cosyvoice":
        print(f"音色: {cfg.tts.cloud_voice_id or '(未注册,会报错)'} @ {cfg.tts.cloud_model}")
    elif cfg.tts.backend == "gpt_sovits":
        print(f"音色: 本地克隆 {cfg.tts.ref_audio}(参考音频决定音色)")
    else:
        print(f"音色: {cfg.tts.voice}  语速: {cfg.tts.rate}")
    print("合成中…")
    t0 = time.monotonic()
    data, sr = tts.synthesize(text or "你好,我是小mo,很高兴认识你。")
    print(f"合成完成: {len(data) / sr:.2f}s 音频,{sr}Hz,耗时 {time.monotonic() - t0:.2f}s")
    if tts.last_note:
        print(f"⚠️ 降级说明: {tts.last_note}(这句不是克隆音色)")
    tts.play(data, sr)
    print("\n听不到声音的话:")
    print("  * 如果 output_device 是虚拟声卡 —— 那是对的,声音进了 VRChat,不是给你听的。")
    print("    想听效果就临时把 config.yaml 里 output_device 改成 null。")
    return 0


def cmd_chat(cfg) -> int:
    """纯文字对话,验证大脑 + 流式切句,不碰音频。"""
    from src.llm import ChatSession

    try:
        chat = ChatSession(cfg.llm, cfg.persona)
    except RuntimeError as exc:
        print(f"\n❌ {exc}\n")
        return 1

    print(f"\n模型: {cfg.llm.model} @ {cfg.llm.base_url}")
    print("输入文字对话,/reset 清空历史,/quit 退出。\n")

    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/quit":
            return 0
        if user == "/reset":
            chat.reset()
            print("(历史已清空)")
            continue

        t0 = time.monotonic()
        first = True
        try:
            # 逐句打印,每句带时间戳 —— 这样能直观看到「首句延迟」
            for sentence in chat.reply(user):
                dt = time.monotonic() - t0
                tag = f"+{dt:.2f}s" if first else f" +{dt:.2f}s"
                print(f"小mo {tag} > {sentence}")
                first = False
        except Exception as exc:
            print(f"[出错] {type(exc).__name__}: {exc}")
            continue

        if first:
            print(f"小mo > (空回复, finish_reason={chat.last_finish_reason})")
        print()


def cmd_asr(cfg) -> int:
    """录一句,看转写结果。用来单独调 VAD 门限。"""
    from src.asr import Transcriber
    from src.audio.capture import Capture
    from src.audio.vad import EnergyVAD

    cap = Capture(cfg.audio.capture_device, cfg.audio.block_ms, cfg.audio.preferred_rate)
    vad = EnergyVAD(cfg.vad, cap.sample_rate, cap.block_size)
    asr = Transcriber(cfg.asr)

    print(f"耳朵: {cap.name} @ {cap.sample_rate}Hz")
    print(f"能量门限: {cfg.vad.energy_threshold}  静音判定: {cfg.vad.silence_ms}ms")
    print("加载模型…")
    asr.warmup()
    print("开始说话(说一句就自动结束),Ctrl+C 退出。\n")

    with cap:
        try:
            while True:
                block = cap.queue.get()
                utt = vad.push(block)
                if utt is None:
                    continue
                dur = len(utt) / cap.sample_rate
                t0 = time.monotonic()
                text = asr.transcribe(utt, cap.sample_rate)
                print(f"[{dur:.2f}s 音频 → {time.monotonic() - t0:.2f}s] {text}\n")
        except KeyboardInterrupt:
            print("\n退出。")
    return 0


def cmd_run(cfg) -> int:
    from src.pipeline import Pipeline

    for w in validate(cfg):
        print(f"⚠️   {w}")
    Pipeline(cfg).run()
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="VRChat AI 虚拟伙伴",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", default=None, help="配置文件路径(默认 config.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="列出所有音频设备(含回环)")
    sub.add_parser("check", help="检查配置")
    sub.add_parser("ear", help="耳朵电平表(验证回环采到声音)")
    sub.add_parser("chat", help="纯文字对话(不碰音频)")
    sub.add_parser("asr", help="录一句测试转写")
    sub.add_parser("run", help="跑全流程")

    p = sub.add_parser("osc", help="测试 OSC 通路")
    p.add_argument("text", nargs="?", default="", help="要发的文字")

    p = sub.add_parser("tts", help="试听 TTS 音色")
    p.add_argument("text", nargs="?", default="", help="要念的文字")

    args = parser.parse_args()
    cfg = load(args.config)

    table = {
        "devices": lambda: cmd_devices(cfg),
        "check": lambda: cmd_check(cfg),
        "ear": lambda: cmd_ear(cfg),
        "osc": lambda: cmd_osc(cfg, args.text),
        "tts": lambda: cmd_tts(cfg, args.text),
        "chat": lambda: cmd_chat(cfg),
        "asr": lambda: cmd_asr(cfg),
        "run": lambda: cmd_run(cfg),
    }
    try:
        return table[args.cmd]()
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
