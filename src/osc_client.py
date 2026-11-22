"""VRChat OSC 客户端 —— AI 的「嘴和脸」。

只做发送(VRChat 收在 9000)。三类地址:
  /chatbox/input       头顶聊天框字幕
  /chatbox/typing      打字中提示
  /avatar/parameters/*  Avatar 动画参数(表情、口型、手势)

⚠️ 关于 /input/*(移动、跳跃、菜单):本项目**默认不碰** —— 那属于「自动操控角色」,
   既容易被判定为自动化,也不是这个项目要解决的问题。
   🔓 **唯一的例外是移动指令**,而且它必须同时满足两个条件:
     ① 说话人**被声纹判定为主人**(`src/audio/voiceprint.py`)
     ② 过了管线的**权限闸门**(`pipeline.py::_gate_permission`)
   别人说"往前走"不会产生任何 /input 写入 —— 见 `input_axis()`。

用法:
    osc = VRChatOSC(cfg)
    osc.chatbox("你好呀")
    osc.param("Viseme", 0.8)
"""

from __future__ import annotations

import threading
import time

from pythonosc.udp_client import SimpleUDPClient

from .config import Config


class VRChatOSC:
    def __init__(self, cfg: Config):
        self._client = SimpleUDPClient(cfg.osc.host, cfg.osc.port)
        self._chatbox_cfg = cfg.chatbox
        self._lock = threading.Lock()
        self._last_chatbox_at = 0.0

    # ---------------- 聊天框 ----------------

    def chatbox(self, text: str, *, immediate: bool = True, force: bool = False) -> bool:
        """把文字发到头顶聊天框。

        受 VRChat 限流约束(chatbox.min_interval_s)。返回 False 表示被限流跳过。
        force=True 用于「最后一条必须发出去」的场合(比如回复结尾)。
        """
        if not self._chatbox_cfg.enabled or not text.strip():
            return False

        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_chatbox_at < self._chatbox_cfg.min_interval_s:
                return False
            self._last_chatbox_at = now

        text = text.strip()[: self._chatbox_cfg.max_chars]
        args: list = [text, immediate]
        if self._chatbox_cfg.send_sound:
            args.append(True)

        self._client.send_message("/chatbox/input", args)
        return True

    def input_axis(self, addr: str, value) -> None:
        """发 `/input/*` —— **移动专用**。

        ⚠️ 这是本项目对"不碰 /input/*"那条规矩的**唯一例外**,而且调用方
        (pipeline 的移动执行)**必须先过权限闸门**。别在别处调它。
        `/input/*` 都是布尔开关式的:按下即动,必须自己再发一次关掉,
        否则角色会一直走 —— 停不下来比不动更糟。
        """
        try:
            self._client.send_message(addr, value)
        except Exception:
            pass          # OSC 发不出去绝不能让语音崩

    def typing(self, on: bool) -> None:
        """聊天框左上角的「正在输入」提示。"""
        if self._chatbox_cfg.enabled:
            self._client.send_message("/chatbox/typing", bool(on))

    # ---------------- Avatar 参数 ----------------

    def param(self, name: str, value) -> None:
        """发一个 Avatar 参数。name 不带 /avatar/parameters/ 前缀。"""
        self._client.send_message(f"/avatar/parameters/{name}", value)

    def params(self, mapping: dict) -> None:
        for k, v in mapping.items():
            self.param(k, v)

    # ---------------- 口型 ----------------

    def mouth(self, openness: float) -> None:
        """嘴巴开合 0.0~1.0。需要 Avatar 里有同名 float 参数 + Animator 层。

        如果你的 Avatar 用的是 Viseme 整数(0-14),把这里换成
        self.param("Viseme", int(openness * 14)) 之类。
        """
        self.param("MouthOpen", max(0.0, min(1.0, float(openness))))
