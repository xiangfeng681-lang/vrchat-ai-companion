"""流水线编排:把耳朵、脑子、嗓子串成一条线。

线程结构(三条):

  [音频采集线程]          pyaudiowpatch 回环 / sounddevice,只做拷贝入队,绝不阻塞
          ↓ queue
  [_listen_loop 线程]     取块 → VAD 切句 → 重采样 → Whisper 转写 → DeepSeek 流式 → 入 TTS 队列
          ↓ queue
  [_speak_loop 线程]      取句子 → TTS 合成 → 播放到虚拟声卡 → 顺带驱动口型

为什么这么切:转写和 LLM 都是阻塞的,放主线程会拖垮音频采集。
TTS 单独一条线程是为了让「LLM 还在生成第 2 句时,第 1 句已经在念了」。

采样率:回环设备强制 48000Hz,所以 VAD 和块大小都按采集的真实率算,
只在送进 Whisper 之前重采样到 16k。

⚠️ 不做打断(barge-in):AI 说话期间不听话。这是刻意的取舍 ——
   回声消除没做好之前,能打断比不能打断更烦人。
"""

from __future__ import annotations

import queue
import re
import socket
import threading
import time

import numpy as np

from .asr import SIMPLIFY, Transcriber
from .audio.capture import Capture
from .audio.vad import EnergyVAD, rms
from .config import Config
from .llm import ChatSession
from .memory import VectorMemory
from .osc_client import VRChatOSC
from .tts import TTS
from .vision.scene import (
    _OWNER_NOTE,
    SceneDescriber,
    find_owner,
    owner_in_plates,
)
from .vision.speaking import PlateTracker, SpeakerTracker

# LLM 在回复里吐的动作标记,如 [挥手] / 【点头】。
#
# 为什么用**括号**而不是自然语言:自然语言分不开"动作"和"描述" ——
# 「我挥了挥手」到底是让它挥手,还是它在叙述?括号是硬边界,不会误判。
# 长度卡 1~12 字,免得把正文里正常的方括号内容(比如 [笑])之外的东西吞掉。
_GESTURE_RE = re.compile(r"[\[【]\s*([^\[\]【】]{1,12})\s*[\]】]")


def split_gesture(text: str) -> tuple[str, str | None]:
    """把句子里的动作标记摘出来,返回 (干净的正文, 动作名或 None)。

    ⚠️ **摘掉标记是必须的** —— 不摘的话 TTS 会把「挥手」两个字念出来。
    """
    m = _GESTURE_RE.search(text)
    if not m:
        return text, None
    return _GESTURE_RE.sub("", text).strip(), m.group(1).strip()


class Pipeline:
    def __init__(self, cfg: Config, *, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose

        self.osc = VRChatOSC(cfg)
        self.chat = ChatSession(cfg.llm, cfg.persona)
        self.tts = TTS(cfg.tts, cfg.audio.output_device)
        self.asr = Transcriber(cfg.asr)
        self.capture = Capture(
            cfg.audio.capture_device, cfg.audio.block_ms, cfg.audio.preferred_rate
        )
        # VAD 必须按采集的真实采样率构造(回环是 48k,不是配置里写的那个)
        self.vad = EnergyVAD(self.cfg.vad, self.capture.sample_rate, self.capture.block_size)

        # 队列里装的是 (句子, 聊天框全文, 本轮代号) 三元组。
        # 句子为 None 表示本轮结束。**代号是关键** —— 让路/打断之后,靠它把
        # 上一轮残留的句子和结束标记认出来丢掉(见 _speak_loop 开头)。
        self._sentences: queue.Queue[tuple] = queue.Queue()
        self._speaking = threading.Event()   # AI 正在说话
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_wake = 0.0                # 上次被叫到名字的时刻(追问窗口用)

        # 向量记忆。关了就是 None,后面所有用到的地方都判空 —— 不装模型、
        # 不建库,零开销。
        self.memory = VectorMemory(cfg.memory) if cfg.memory.enabled else None

        # 「闭嘴」打断用的状态。说话期间走一条**旁听通道**:音频不进 VAD,
        # 而是攒起来定期转写,只为了听出停止词。
        self._barge = threading.Event()      # 置位 = 立刻掐断当前播放
        self._barge_buf: list[np.ndarray] = []
        self._barge_silence = 0
        # 旁听通道的**第二个产出**:听到的不是停止词,但确实是人在说话。
        # 由说话线程在**句与句之间**消费 —— 别人开口了就把剩下的话让出去。
        self._heard_other = threading.Event()

        # 动作联动:发给 tools/pose_server.py 的指令端口。
        # 用**自己一个 socket**,不复用 VRChatOSC —— 收件方根本不是同一个。
        # gesture_port=0 就是关掉联动。
        self._gesture_sock = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                              if cfg.behavior.gesture_port else None)

        # 当前累计的转向角度(度)。**必须累加,不能每句重算** ——
        # 见 `_look_at_speaker` 里的说明:转过之后人在画面正中,
        # 重算会得到 0°,把刚转的又撤销回去。
        self._look_yaw = 0.0
        self._look_sent_at = 0.0

        # 「谁在说话」—— 靠名牌发光描边的视觉判断(见 src/vision/speaking.py)。
        # ⚠️ **整个视觉链路都是可选的**:找不到 VRChat 窗口、抓不到帧、
        #    OCR 挂了,都必须只让"说话人 = 未知",绝不能影响语音。
        #    `start()` 返回 False 是正常情况,不是错误。
        self.speaker = SpeakerTracker(title=cfg.behavior.vision_window)

        # 名牌位置(名字 + 在画面里的横向位置)—— 配合音频方向,回答
        # "左边那个正在说话的人是谁"。同样**可以起不来**。
        self.plates = PlateTracker(title=cfg.behavior.vision_window)

        # 「这声音是不是主人」—— 声纹,和上面那个视觉的 `speaker` 是**两条独立的路**。
        #   视觉 → 知道画面里谁在说,但只在画面内
        #   声纹 → **背对着、不在画面里也认得出**,但只认主人一个人
        # 用途是权限分级(主人能下移动指令,别人只能动手动脚/跳舞/转视角)。
        # ⚠️ 和其它视觉件一样**可失败**:没注册、模型丢了、音频太短,
        #    一律退化成"不是主人",绝不影响语音本身。
        self.voiceprint = None
        if cfg.voiceprint.enabled:
            try:
                from src.audio.voiceprint import Voiceprint
                vp = Voiceprint(cfg.voiceprint)
                if vp.ready:
                    self.voiceprint = vp
                    print("声纹: 已启用(主人声纹已注册)", flush=True)
                else:
                    print("声纹: 未启用(缺模型或缺 owner.npy —— 先跑 tools/enroll_voice.py)",
                          flush=True)
            except Exception as exc:
                print(f"声纹: 未启用({type(exc).__name__}: {exc})", flush=True)

        # 「看得见」—— 截图发给视觉大模型,拿回一句描述注入对话。
        # **走旁路**:主链路还是 deepseek-chat,视觉是异步的、可失败的。
        # token 账打进日志 —— 视觉是**按次计费**的,看不到消耗就没法判断成本。
        # 跳过(画面没变)不打印,不然刷屏。
        def _scene_report(used: int, total: int, dt: float, skipped: bool) -> None:
            if skipped or not used:
                return
            self.log("眼睛", f"看了 {dt:.1f}s  本次 {used} tokens / 累计 {total}")

        self.scene = SceneDescriber(cfg.scene, llm_cfg=cfg.llm, on_report=_scene_report)
        # 每轮对话的"代号"。被打断时 +1,大脑线程发现代号变了就不再往下生成 ——
        # 否则嗓子停了、脑子还在往队列里塞,停不干净。
        self._epoch = 0

    # ---------------- 打断(停止词) ----------------

    def _match_stop(self, text: str) -> bool:
        words = self.cfg.wake.stop_words or []
        if not words:
            return False
        # ⚠️ 先繁简归一。auto 模式下 ASR 会吐繁体,不归一的话「閉嘴」匹配不到
        #    词表里的「闭嘴」—— 一个字都对不上,打断直接失灵。
        squashed = "".join(
            ch for ch in text.translate(SIMPLIFY)
            if not ch.isspace() and ch not in "，。！？、,.!?~～"
        ).lower()
        return any(w.lower() in squashed for w in words if w)

    def _barge_push(self, block: np.ndarray) -> None:
        """AI 说话期间的旁听:攒够一段就转写一次,找停止词。

        只在**有声音**的时候攒 —— 不然它自己念 10 秒,我们就白跑十几次 Whisper。
        """
        if rms(block) < self.cfg.vad.energy_threshold:
            self._barge_silence += 1
            if self._barge_silence > 6:      # 连续约 180ms 安静就丢掉缓冲
                self._barge_buf.clear()
            return
        self._barge_silence = 0
        self._barge_buf.append(block)

        need = int(self.capture.sample_rate * self.cfg.wake.stop_check_s)
        if sum(len(b) for b in self._barge_buf) < need:
            return

        audio = np.concatenate(self._barge_buf)
        self._barge_buf.clear()
        try:
            # **必须走 quick** —— 这条通道每 stop_check_s 就跑一次,用完整
            # 配置(beam=5 + 提示词)会把采集线程占满,真正该转写的话得排队,
            # 实测把一次普通转写拖到 3~8 秒。
            text = self.asr.transcribe(audio, self.capture.sample_rate, quick=True)
        except Exception:
            return
        if not text:
            return
        if self._match_stop(text):
            self.log("打断", f"听到停止词: {text!r}")
            self._barge.set()
            return
        # 不是停止词,但确实是人在说话 —— 记下来,让说话线程在下一句之前让路。
        # ⚠️ **必须过 looks_like_speech**:quick 通道本身不过幻觉闸,不把关的话
        #    B站/音乐里编出来的媒体片尾会被当成"别人在说话",让它一直让路。
        if self.cfg.behavior.yield_to_others and self.asr.looks_like_speech(text):
            self._heard_other.set()

    def _gate_taunt(self, sentence: str, heard: str) -> str:
        """「招牌反击」硬闸 —— 没被挑衅就不许说。

        🔴 为什么必须用代码拦:人设里写了「只在被挑衅之后说,这条没有例外」,
        **但它还是会不时地冒出来**。提示词是"劝",劝不住就得有人动手 ——
        和 `max_sentences` 那道闸同一个道理。

        做法很直白:用户这句话里没有挑衅词,就把招牌句从回复里剥掉。
        **只剥招牌句,不整句丢掉** —— 剩下的内容通常是正常的,照念。
        """
        word = self.cfg.behavior.taunt_reply
        if not word or word not in sentence:
            return sentence
        # 用户这句话里出现过挑衅词吗?出现过就是正当反击,放行。
        low = (heard or "").lower()
        if any(w and w.lower() in low for w in self.cfg.behavior.taunt_words):
            return sentence
        cleaned = sentence.replace(word, "")
        # 剥完可能只剩标点和波浪号,那就别念了
        cleaned = re.sub(r"^[\s~～\-—、,，。.!！?？]+", "", cleaned).strip()
        if len(cleaned) < 2:
            self.log("招牌句", f"没被挑衅,整句丢掉: {sentence!r}")
            return ""
        self.log("招牌句", f"没被挑衅,剥掉「{word}」: {sentence!r} → {cleaned!r}")
        return cleaned

    # ---------------- 权限闸门 ----------------

    def _gate_permission(self, gesture: str | None, is_owner: bool) -> str | None:
        """**移动类动作只有主人能触发。** 返回允许执行的 gesture(可能是 None)。

        ## 为什么必须用代码拦

        和 `_gate_taunt`、`max_sentences` 同一个道理:**提示词是"劝"**。
        人设里会写"别人让你走的时候不要动",但公开世界里那句话是陌生人说的,
        靠提示词拦等于把"小mo 会不会被陌生人带走"押在模型的自觉上。
        **权限这种事必须在代码层动手。**

        ## 判据

        `is_owner` 来自**声纹**(`src/audio/voiceprint.py`)——
        不是视觉的发光描边。声纹的好处是**背对着、不在画面里也认得出**,
        代价是**只认主人一个人**,而且认不出时一律当"非主人"。

        ## 剥掉之后

        只剥**标记**,不整句丢 —— 剩下的正文(比如「才不听你的呢」)照念。
        这跟 `_gate_taunt` 的策略一致:角色该有的反应留着,不该做的动作掐掉。
        """
        if not gesture:
            return gesture
        if is_owner:
            return gesture
        if gesture in (self.cfg.behavior.move_actions or []):
            self.log("权限", f"❌ 非主人,拒绝移动指令「{gesture}」")
            return None
        return gesture          # 动作/跳舞/转视角对谁都放行

    def _fire_move(self, name: str) -> None:
        """执行移动指令(走 OSC `/input/*`)。

        ⚠️ **只在过了权限闸门之后被调用。** 路径是:
            声纹判定 is_owner → `_gate_permission` 放行 → 这里

        `/input/*` 是**按住的语义**:发一次 True 只是"按下",必须再发一次
        False 松开,否则人物会一直朝那个方向走 —— **停不下来比不动更糟**,
        所以松开那一下放在 `finally` 里,连异常路径都要执行。
        """
        ADDR = {
            "前进": "/input/MoveForward", "后退": "/input/MoveBackward",
            "左移": "/input/MoveLeft", "右移": "/input/MoveRight",
        }
        if name in ("跟随", "过来", "停下"):
            # 这三个交给**跟随服务**(常驻进程,自己知道主人在哪、多远停)。
            # 它和 `/input/*` 是两条不同的路:方向键走 OSC 直接发,这三个走
            # UDP 指令让服务自己决策。
            port = int(getattr(self.cfg.behavior, "follow_port", 0) or 0)
            if not port or not self._gesture_sock:
                self.log("移动", f"「{name}」:跟随服务没开(follow_port=0),这次只记录")
                return
            cmd = "stop" if name == "停下" else "follow"
            try:
                self._gesture_sock.sendto(cmd.encode("utf-8"), ("127.0.0.1", port))
                self.log("移动", f"→ 跟随服务: {cmd}")
            except Exception as exc:
                self.log("移动", f"跟随服务指令失败 {type(exc).__name__}: {exc}")
            return
        addr = ADDR.get(name)
        if not addr:
            return
        secs = float(self.cfg.behavior.move_step_s)
        self.log("移动", f"▶ {name}  ({secs:.2f}s)")
        try:
            self.osc.input_axis(addr, True)
        except Exception:
            return
        # 🔴 **不能在这里 sleep 等松开。** 这个函数是从 `on_ready` 回调里调的,
        #    而那是"音频即将出声"的那一刻 —— 阻塞 0.45 秒会把播放卡住。
        #    所以按下之后交给定时器去松手,自己立刻返回。
        t = threading.Timer(secs, self._release_axis, args=(addr,))
        t.daemon = True
        t.start()

    def _release_axis(self, addr: str) -> None:
        """松开方向键。**跑在定时器线程里**,异常同样要吞干净。"""
        try:
            self.osc.input_axis(addr, False)
        except Exception:
            pass

    def _fire_gesture(self, name: str | None) -> None:
        """把动作名发给动作播放器。

        ⚠️ **异常必须全部吞掉**:动作播放器没开着是很正常的情况
        (它是个独立进程),绝不能因为动作发不出去就把说话也搞崩。
        """
        # 临时诊断:确认这个回调到底有没有被调到、收到的名字是什么。
        # (怀疑 on_ready 在某些路径下不触发 —— 见 _fire_move 的日志缺失)
        self.log("点火", f"_fire_gesture({name!r})  "
                        f"移动类={name in (self.cfg.behavior.move_actions or [])}")
        if name in (self.cfg.behavior.move_actions or []):
            self._fire_move(name)
            return
        self._send_pose(name)

    def _send_pose(self, text: str | None) -> None:
        """给动作播放器发一条指令(动作名,或 `look:<度数>`)。

        ⚠️ **异常必须全部吞掉**:播放器没开着是很正常的情况(它是独立进程),
        绝不能因为发不出去就把说话也搞崩。
        """
        if not self._gesture_sock or not text:
            return
        try:
            self._gesture_sock.sendto(
                text.encode("utf-8"),
                ("127.0.0.1", int(self.cfg.behavior.gesture_port)))
        except Exception:
            pass

    def _look_at_speaker(self, t1: float, t2: float) -> None:
        """听到有人说话 → **转头看向他**。

        方向怎么来的:
          ① **左右靠立体声** —— VRChat 的玩家语音是空间音频渲染的,人在左边
             声音就偏左(见 `audio/capture.py::direction`)。这是"谁在说话"
             的替代信号 —— 视觉那条发光描边的路一直不稳。
          ② **是谁靠 OCR 读名牌** —— 知道"左边那个名牌"在哪,才知道该转多少度。

        ⚠️ **只能分左右,分不了前后。** 两声道下正前方和正后方的左右能量
        完全相同;人在背后时会退化成"看着正前方"。要分前后得再叠视觉
        (人在不在画面里),目前没做。

        ⚠️ **整条链路都可失败**:抓不到图、pose_server 没开、音频没方向 ——
        任何一种都只是"不转",绝不影响说话。
        """
        if not self.cfg.behavior.look_at_speaker:
            return
        if not self.plates.ok:
            self.log("转向", "跳过:名牌定位没起来")
            return
        try:
            # 取**最后一秒**的方向,不是整句话的平均 ——
            # 人说话时经常边走边说,整句的平均位置没意义,
            # **说完那一刻他在哪**才是该看的。实测踩过:整句平均判成"左",
            # 但名牌明明在右边。
            side = self.capture.direction(max(t1, t2 - 1.2), t2)
            plates = self.plates.plates()

            # 🔴 **角度符号是实测标定的,别按"直觉"改。**
            # 本来按右手系推导写成 `(cx-0.5)*fov`(名牌在左 → 负角),**转反了**。
            # 实测:发 `look:+35` 之后名牌 cx 从 0.142 移到 0.488 ——
            # **物体右移 = 镜头向左转**,即**这个系统里正角 = 左转**。
            # 所以名牌在左(cx<0.5)要给**正**角;声音偏右(side>0)要给负角。
            FOV = self.cfg.behavior.look_fov

            if plates:
                if len(plates) == 1:
                    # 🔑 **只有一个人时不需要问音频。** 画面里就他一个,
                    # 说话的不是他是谁?音频只在**多人**时才用来区分是哪一边那个。
                    plate, how = plates[0], "画面里只有他一个"
                elif side is None:
                    self.log("转向", f"跳过:多人但没测到方向  (名牌{len(plates)}个)")
                    return
                else:
                    plate = self.plates.pick(side, deadzone=self.cfg.behavior.look_deadzone)
                    if plate is None:
                        self.log("转向", f"跳过:那一侧没名牌  (方向{side:+.3f},"
                                         f" 名牌{len(plates)}个)")
                        return
                    how = f"声音偏{'右' if side > 0 else '左'} {side:+.3f}"
                delta = (0.5 - plate["cx"]) * FOV
                what = plate["raw"]
                extra = f"cx={plate['cx']:.2f} 还差{delta:+.0f}°"
            elif side is not None and abs(side) >= self.cfg.behavior.look_deadzone \
                    and self.cfg.behavior.look_audio_gain:
                # **画面里看不见他,但声音知道在哪边** —— 大致转过去。
                # 这补的是视觉的死角:人在侧后方说话,名牌在画面外,
                # 而那恰恰是最该转身的情况。精度差,但比一动不动强。
                delta = -side * self.cfg.behavior.look_audio_gain
                what, how = "(视野外)", f"只靠声音 {side:+.3f}"
                extra = "无名牌"
            else:
                self.log("转向", f"跳过:画面里没人,声音也没方向"
                                 f"(方向{None if side is None else round(side,3)})")
                return

            # 🔑 **累加,不是重算。** pose_server 那边的角度是**绝对的**,
            # 而人在画面里的位置是**相对于当前朝向**的。转过之后人会跑到
            # 画面正中(cx≈0.5),这时候如果按 `(0.5-cx)*FOV` 直接当目标发,
            # 得到的是 0° —— **等于把刚转的撤销回去**。
            # 实测症状:说完一句就转回正面,再说话又转过去,来回点头。
            #
            # 所以这里把 `(0.5-cx)*FOV` 当成**还差多少**(闭环误差),
            # 累加到当前朝向上。人会越转越靠近画面中心,自然收敛。
            now = time.monotonic()
            if now - self._look_sent_at > self.cfg.behavior.look_hold_s:
                # 这么久没发过指令,pose_server 已经自动回正了 —— 我们这边也得归零,
                # 否则基准不一致,下一句会从错误的角度开始累加。
                self._look_yaw = 0.0
            self._look_yaw = max(-60.0, min(60.0, self._look_yaw + delta))
            target = self._look_yaw
            self._send_pose(f"look:{target:+.1f}")
            self._look_sent_at = now
            self.log("转向", f"{what}  ({how}, {extra} → 累计 {target:+.0f}°)")
        except Exception as exc:
            # ⚠️ **别吞掉原因** —— 之前这里写 `except: pass`,结果"没转向"
            # 排查时一行线索都没有,只能靠猜。宁可吵一点也要把原因打出来。
            self.log("转向", f"出错 {type(exc).__name__}: {exc}")

    def _drain_sentences(self) -> int:
        """丢掉队列里还没念的句子,返回丢了几句。"""
        n = 0
        while True:
            try:
                self._sentences.get_nowait()
                n += 1
            except queue.Empty:
                return n

    # ---------------- 唤醒词 ----------------

    def _strip_wake(self, text: str) -> tuple[bool, str]:
        """看这句话里有没有叫名字,返回 (是否被叫到, 去掉称呼后的正文)。

        ASR 的输出很随意,所以匹配前先把空格和标点抹掉 ——
        「小 mo」「小莫,」「小莫。」都得认出来。抹掉的只是匹配用的副本,
        正文返回时保留原样。
        """
        import re

        words = self.cfg.wake.words or []
        # 繁简归一后再匹配(见 SIMPLIFY 的说明)。
        # 唤醒词本身繁简同形(小莫/小茉/小墨),理论上不受影响 —— 但正文也要用
        # 归一后的,免得以后往词表里加繁体敏感的词时,这里悄悄失配。
        text_s = text.translate(SIMPLIFY)
        # 匹配用的副本:抹掉空格和标点、统一小写。
        # 统一小写是必须的 —— ASR 会吐 "小MO"/"小Mo"/"小mo",配置里不可能穷举。
        squashed = "".join(
            ch for ch in text_s if not ch.isspace() and ch not in "，。！？、,.!?~～"
        ).lower()
        for word in words:
            key = "".join(ch for ch in word if not ch.isspace()).lower()
            if not key or key not in squashed:
                continue
            # 摘掉正文里的称呼,免得 AI 对着"小莫"这两个字本身回复。
            # 正则允许字之间夹空格、并忽略大小写,这样「小 mo」「小Mo」都能摘干净。
            pattern = r"\s*".join(re.escape(c) for c in key)
            body = re.sub(pattern, "", text_s, count=1, flags=re.IGNORECASE)
            return True, body.strip(" ,，。！？、.!?~～")
        return False, text

    def _wake_gate(self, text: str) -> str | None:
        """唤醒词闸门。返回该送进大脑的正文;返回 None 表示这句不关它的事。

        没有唤醒词(默认)时直接放行,行为和以前完全一样。
        开了唤醒词之后才有三种情况:叫了名字 / 还在追问窗口内 / 没叫 ——
        最后一种丢掉,这就是它不再"插嘴"的原因。
        """
        w = self.cfg.wake
        if not w.enabled:
            return text

        if len(text) < w.min_chars:
            self.log("忽略", f"太短: {text!r}")
            return None

        called, body = self._strip_wake(text)
        now = time.monotonic()

        if called:
            self._last_wake = now
            if not body:
                if not w.reply_to_name_only:
                    self.log("忽略", f"只叫了名字: {text!r}")
                    return None
                # 只喊了一声名字也要应 —— 否则用户会觉得"我叫它它不理我"
                self.log("唤醒", f"{text}  (只叫了名字)")
                return "（用户叫了你一声，回应一下）"
            self.log("唤醒", f"{text}  → 正文: {body!r}")
            return body

        # ⚠️ `followup_s > 0` 这个前置判断**不能省**。
        # 它表达的是"窗口关闭",而如果只写 `now - self._last_wake <= 0`,在
        # Windows 上会**失效**:`time.monotonic()` 精度约 15.6ms,紧挨着的两次
        # 调用会返回**同一个值**,差正好是 0.0,`0 <= 0` 为真 ——
        # 于是"必须叫名字"被悄悄放水成"跟上一句同一时刻说的就放行"。
        # (实测踩到:设了 0 之后「早上好」照样被放行。)
        if w.followup_s > 0 and now - self._last_wake <= w.followup_s:
            # **滑动窗口**:你说一句,窗口就往后顺延。
            # 不刷新的话窗口从"叫名字那一刻"起算,聊到第 16 秒会突然不理人 ——
            # 体感是"聊着聊着它掉线了",很怪。
            # 代价:只要一直有人说话它就保持清醒,所以窗口别设太长。
            self._last_wake = now
            # 用不同的标签 —— 之前和主流程都打「听到」,同一句在日志里出现
            # 两次,看着像"处理了两遍"。
            self.log("在谈", f"{text}  (窗口顺延 {w.followup_s:.0f}s){self._lang_note()}")
            return text

        self.log("忽略", f"{text}  (没叫名字){self._lang_note()}")
        return None

    # ---------------- 日志 ----------------

    def log(self, tag: str, msg: str) -> None:
        if self.verbose:
            print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)

    def _lang_note(self) -> str:
        """自动语种识别的结果,拼成一段可直接贴进日志的尾巴。

        **只在非中文时返回内容** —— 中文是常态,每句都打会把日志淹掉;
        而"英文/日文被认成了什么"才是需要一眼看到的信息(见 config 的 asr.language)。
        """
        lang = getattr(self.asr, "last_lang", "")
        if not lang or lang == "zh":
            return ""
        prob = getattr(self.asr, "last_lang_prob", 0.0)
        return f"  / 语言={lang}({prob:.2f})"

    # ---------------- 生命周期 ----------------

    def run(self) -> None:
        self.log("启动", f"耳朵 = {self.capture.name} @ {self.capture.sample_rate}Hz")
        self.log("启动", f"嗓子 = {self.tts.name}")
        self.log("启动", f"嘴巴 = osc {self.cfg.osc.host}:{self.cfg.osc.port}")
        self.log("启动", f"大脑 = {self.cfg.llm.model} @ {self.cfg.llm.base_url}")

        # ⚠️ **云端后端不能预热本地模型** —— 那会把 Whisper 加载进显存,
        #    "换云端省 2G 显存"这个目的就白搭了。踩过:改成云端之后显存只降了 1.1G,
        #    查下来就是这行无条件预热干的。
        if getattr(self.cfg.asr, "backend", "local") == "cloud":
            self.log("预热", f"语音识别走云端({self.cfg.asr.cloud_model}),"
                             f"不加载本地模型")
        else:
            self.log("预热", f"加载 Whisper {self.cfg.asr.model} …(第一次会下载模型)")
            self.asr.warmup()
        if self.memory is not None:
            n = self.memory.count()
            self.log("启动", f"记忆 = {n} 条 ({self.memory.db_path.name})")
            self.memory.warmup()     # 提前加载嵌入模型,免得第一轮多等一秒
        # 「谁在说话」的视觉跟踪。**起不来是正常情况**(没开 VRChat、窗口标题不对、
        # WGC 被占),降级成"说话人未知"就行,绝不能因此不跑语音。
        #
        # ⚠️ 默认**关** —— 它从来没可靠过,而且功能已被"立体声判左右"替代,
        #    开着纯属白烧 CPU(每 3 秒一次 OCR)。详见 config 的 `speaker_track`。
        if not self.cfg.behavior.speaker_track:
            self.log("启动", "说话人识别 = 关(视觉那条从来不可靠,见 config 注释)")
        elif not self.cfg.behavior.vision_window:
            self.log("启动", "说话人识别 = 关(配置里没给窗口)")
        elif self.speaker.start():
            self.log("启动", f"说话人识别 = 盯「{self.cfg.behavior.vision_window}」")
        else:
            self.log("启动", f"说话人识别 = 未启用(找不到"
                             f"「{self.cfg.behavior.vision_window}」窗口)")

        # 名牌位置 —— 配合音频方向做"转向说话的人"。起不来就不转。
        if self.cfg.behavior.look_at_speaker and self.plates.start():
            self.log("启动", "名牌定位 = 开(配合立体声做转向)")
        elif self.cfg.behavior.look_at_speaker:
            self.log("启动", "名牌定位 = 未启用 → 不会转向说话的人")

        # 「看得见」的旁路。同样**起不来是正常的** —— 没配模型、没 key、
        # 模型下线(它是 Exp)都算,降级成"看不见"就行。
        if self.scene.start():
            self.log("启动", f"眼睛 = {self.cfg.scene.model} "
                             f"(只在唤醒时看,限流 {self.cfg.scene.min_gap_s:.0f}s,"
                             f"最多等 {self.cfg.scene.wait_s:.1f}s)")
        else:
            self.log("启动", "眼睛 = 未启用 → 小mo 看不见画面")

        self.log("预热", "完成")

        self.osc.chatbox("我上线啦~", force=True)

        for target in (self._speak_loop, self._listen_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

        with self.capture:
            try:
                while not self._stop.is_set():
                    time.sleep(0.2)
            except KeyboardInterrupt:
                self.log("退出", "收到 Ctrl+C")
            finally:
                self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self.speaker.close()          # 常驻的 WGC 会话要显式关
        except Exception:
            pass
        try:
            self.scene.close()
        except Exception:
            pass
        try:
            self.plates.close()
        except Exception:
            pass
        try:
            self.osc.chatbox("我先下线啦,回头见。", force=True)
            self.osc.mouth(0.0)
        except Exception:
            pass
        self._sentences.put((None, None, self._epoch, None))
        for t in self._threads:
            t.join(timeout=2.0)

    # ---------------- 耳朵 → 脑子 ----------------

    def _listen_loop(self) -> None:
        while not self._stop.is_set():
            try:
                block = self.capture.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._speaking.is_set():
                # AI 在说话。默认**闭麦**(自激保护),但配了停止词就要开着旁听 ——
                # 否则你根本插不上嘴,只能干等它说完。
                if self.cfg.wake.stop_words:
                    self._barge_push(block)
                else:
                    self.vad.reset()
                continue

            utterance = self.vad.push(block)
            if utterance is None:
                continue

            try:
                self._handle_utterance(utterance)
            except Exception as exc:  # 单轮失败不该让整个进程挂掉
                self.log("错误", f"{type(exc).__name__}: {exc}")

    def _who_is_owner(self, scene: str) -> tuple[str | None, str]:
        """主人现在在不在画面里?返回 `(别名 or None, 判据说明)`。

        ## 为什么优先用名牌列表,而不是视觉描述

        描述是**一大段自然语言**,名字只要出现在**任意一句话**里就算命中。
        而这段文字里混着视觉模型的散文、名牌、以及**小mo 自己的字幕气泡**
        (提示词明确要求"把画面里所有文字一字不差读出来")。于是会**自激**:

            小mo 念「干嘛呀,莫西干。」→ 字幕挂头顶 → 下一轮描述里读到「莫西干」
            → 又判"主人在场" → 继续念「莫西干」

        名牌列表(`PlateTracker`)没这个回路:逐块 OCR、过掉长度(2~20)和
        置信度(>0.5)两道闸,而且要求名字落在**一块牌子内部**。

        ⚠️ **名牌列表可用时以它为准,不命中就是不命中** —— 不再退回去搜描述,
        因为那正是误判的来源。只有 OCR 整个起不来(`plates.ok == False`)时
        才退回描述,免得主人在场却完全认不出。

        判据说明会打进日志(`判据名牌「…」` / `判据描述文本`)——
        认错了要能一眼看出是哪条路认错的,不然只能猜。
        """
        aliases = self.cfg.behavior.owner_names
        if self.plates.ok:
            hit = owner_in_plates(self.plates.plates(), aliases)
            if hit:
                return hit[0], f"名牌「{hit[1]}」"
            return None, "名牌列表"
        return find_owner(scene, aliases), "描述文本(OCR 未启用)"

    def _handle_utterance(self, audio) -> None:
        t0 = time.monotonic()
        dur = len(audio) / self.capture.sample_rate

        # ---- 这段话是谁说的 ----
        # 🔑 **必须按"这段话的时间窗"回头查,不能问"现在谁在发光"。**
        # VAD 要等尾部静音才切句(2~3 秒),等话切出来发光早就灭了。
        # 窗口 = [切句时刻 - 音频时长 - 一点余量, 切句时刻]。
        # 认不出就是 None —— **宁可不知道,也别猜一个安上去**(会串记忆)。
        who = None
        if self.speaker.ok:
            who = self.speaker.speaker_at(t0 - dur - 0.4, t0)

        text = self.asr.transcribe(audio, self.capture.sample_rate)
        if not text:
            # 把**被哪道闸丢的**打出来。之前只写"空结果",一堆空的时候
            # 分不清是静音、是幻觉、还是闸门把真话误杀了 —— 这三种的处理
            # 方式完全不同(前两种无视,第三种要放宽阈值)。
            why = getattr(self.asr, "last_drop", "") or "未知"
            # 「无人声」还带上数值 —— 这是判断"阈值太严"还是"确实没人说"的
            # 唯一依据:0.6~0.7 = 真话很可能被误杀,0.85+ = 真是音乐/噪音。
            ns = getattr(self.asr, "last_nospeech", 0.0)
            extra = f" {ns:.2f}" if why == "无人声" else ""
            self.log("转写", f"空结果[{why}{extra}]  (音频 {dur:.1f}s / "
                            f"耗时 {time.monotonic() - t0:.2f}s{self._lang_note()})")
            return

        # 唤醒词闸门。放在转写**之后** —— 反正每句都要转写,顺手看一眼有没有
        # 叫名字是零成本的;专门搞个唤醒词引擎反而要多一份依赖和一份维护。
        # 没开唤醒词时它是空操作,行为与以前完全一致。
        text = self._wake_gate(text)
        if text is None:
            return

        # ---- 声纹:这段声音是不是主人 ----
        # 放在唤醒闸门**之后** —— 被忽略的话根本不用算,省 CPU。
        # 认不出/出错一律当"不是主人",是安全的那一侧。
        is_owner = False
        if self.voiceprint is not None:
            try:
                is_owner = self.voiceprint.is_owner(audio, self.capture.sample_rate)
                sc = self.voiceprint.last_score
                if sc is None:
                    self.log("声纹", "没听到有效声音,按非主人处理")
                else:
                    verdict = "✅ 主人" if is_owner else "❌ 非主人"
                    # ⚠️ 这个数**不是中位数** —— 是「最好的两窗平均」(top2),
                    #    见 src/audio/voiceprint.py:152 的说明。标签以前写错成
                    #    「中位分数」,看日志的人会据此以为取的是中位数。
                    self.log("声纹", f"{verdict}  top2均值 {sc:.3f}  "
                                    f"(阈值 {self.cfg.voiceprint.threshold})")
            except Exception as exc:      # 声纹绝不拖垮语音
                self.log("声纹", f"失败 {type(exc).__name__}: {exc}")

        # 把音频时长和转写耗时一起打出来 —— 判断"慢"是音频太长还是推理太慢,
        # 这两个原因的处理方式完全不同。
        self.log("听到", f"{text}  (转写 {time.monotonic() - t0:.2f}s / "
                        f"音频 {dur:.1f}s{self._lang_note()})")
        if who:
            self.log("说话人", who)

        # 被叫到了 —— 顺手让它"看"一眼。**异步,不等** —— 视觉要 1~3 秒,
        # 等它就把首句延迟毁了。这一轮多半用不上新画面,下一轮就能用上。
        self.scene.refresh_now()

        # 转向说话的人。放在**唤醒闸门之后** —— 也就是"只在有人对它说话时转",
        # 而不是屋里谁开口都转。后者更像真人(跟着话题转头),但也更容易显得
        # 抽风;要改的话把这行挪到唤醒判断前面就行。
        self._look_at_speaker(t0 - dur - 0.4, t0)

        if self.cfg.behavior.chatbox_echo_user:
            self.osc.chatbox(f"[玩家] {text}", force=True)

        self._speaking.set()          # 从这一刻起不再听自己
        self.osc.typing(True)

        # 向量记忆:按这句的语义去库里捞相关往事。放在调 LLM **之前** ——
        # 那段时间本来就在等网络,嵌入走 CPU 不受影响。
        memories = None
        if self.memory is not None:
            try:
                memories = self.memory.recall(text, speaker=who)
            except Exception as exc:
                self.log("记忆错误", f"{type(exc).__name__}: {exc}")
            if memories:
                self.log("回忆", f"捞到 {len(memories)} 条相关往事")

        # 画面:**最多等 wait_s 秒**看这次"看"能不能赶在这一轮之前完成。
        # ⚠️ 这里绝不能无限等 —— 看图要 3 秒上下,全等下去首句延迟直接翻倍。
        #    等不到就返回上次那份(可能是 None),退化成和以前一样。
        t_wait = time.monotonic()
        scene = self.scene.wait_ready(self.cfg.scene.wait_s) if self.scene.ok else None
        if scene:
            # 「做出你的人」在不在画面里?**由程序判断,不让大模型自己认** ——
            # 视觉读名字会读错(实测 莫西干 → 夏西干/巽西干),它拿到的是错字。
            # 命中就在描述前面加一句明确的,它就不可能漏掉。
            owner, how = self._who_is_owner(scene)
            if owner:
                scene = _OWNER_NOTE + scene
                self.log("看到", f"[主人「{owner}」在画面里 · 判据{how}] "
                                 f"(等待 {time.monotonic() - t_wait:.1f}s)")
            else:
                self.log("看到", f"{scene}  (等待 {time.monotonic() - t_wait:.1f}s)")
        elif self.scene.ok:
            self.log("看到", f"(这轮没赶上,等了 {time.monotonic() - t_wait:.1f}s)")

        display = ""                  # 聊天框累积显示的内容
        full_reply = ""               # 完整回复(存记忆用,不能被 chatbox 上限截断)
        first = True
        count = 0                     # 本轮已经放行了几句
        my_epoch = self._epoch        # 记下本轮代号,被打断时它会变
        self._barge.clear()           # 清掉上一轮可能残留的打断标志
        self._heard_other.clear()     # 让路信号同理:上一轮的不能带到这一轮
        interrupted = False
        try:
            for sentence in self.chat.reply(text, memories=memories,
                                            scene=scene):
                if self._epoch != my_epoch:
                    self.log("打断", "本轮已停止,不再生成后续句子")
                    self.chat.abort()          # 掐断还在流式的 LLM 请求
                    interrupted = True
                    return
                # 先把动作标记摘出来 —— **必须在进 display 和队列之前**,
                # 否则 TTS 会把「挥手」这两个字念出来。
                sentence, gesture = split_gesture(sentence)
                # 🆕 **权限闸门:移动类只有主人能触发**(见 _gate_permission)。
                # 放在动作入队**之前** —— 剥掉的标记不该往下走一步,
                # 否则万一哪条路径漏判,它还是会被执行。
                gesture = self._gate_permission(gesture, is_owner)
                # 招牌句硬闸:没被挑衅就不许说(见 _gate_taunt)
                sentence = self._gate_taunt(sentence, text)
                if not sentence and not gesture:
                    continue                   # 整句只有一个空标记,丢掉
                if sentence:
                    # **句数硬闸**。人设里写了"只说一到两句",但提示词是劝 ——
                    # 实测还是能吐 2~3 句、十来秒,在语音里太长。这里才是拦。
                    # 判断放在累加**之前**:丢掉的那句不该进记忆(用户没听见)。
                    # ⚠️ 只有**真要念的句子**才占额度 —— 纯动作标记不占,
                    #    否则一个 [挥手] 就能把一轮的说话额度吃光。
                    if 0 < self.cfg.behavior.max_sentences <= count:
                        self.log("限长", f"已够 {count} 句,丢掉后续并掐断生成")
                        self.chat.abort()      # 不掐的话脑子会继续读完整段,白烧 token
                        break
                    if first:
                        self.log("首句", f"{sentence}  (+{time.monotonic() - t0:.2f}s)")
                        first = False
                    count += 1
                    full_reply += sentence
                    display = (display + sentence)[-self.cfg.chatbox.max_chars:]
                if gesture:
                    self.log("动作", f"{gesture}  (跟这一句一起出)")
                # 句子、字幕、动作一起塞进队列 —— 都由说话线程在
                # **合成完成、即将出声**那一刻才发出去,这样声音、字幕、
                # 动作三者天然同步。在这里直接发的话会早一个合成的时长。
                # **带上本轮代号**:让路/打断之后,说话线程靠它认出过期句子。
                self._sentences.put((sentence, display, my_epoch, gesture))
            if first:                          # 一句都没生成
                self.log("警告", f"回复为空 (finish_reason={self.chat.last_finish_reason})")
        finally:
            # 本轮结束标记。**也带代号** —— 否则被掐掉那轮的标记会跑到下一轮
            # 才被消费,把新一轮的 _speaking 提前清掉,AI 就开始听见自己说话了。
            self._sentences.put((None, None, my_epoch, None))
            # 记进长期记忆。**被打断的那轮不记** —— 用户听到的是残句,
            # 存进去将来检索出来会莫名其妙。
            if self.memory is not None and not interrupted and full_reply.strip():
                try:
                    self.memory.add(text, full_reply, speaker=who)
                except Exception as exc:
                    self.log("记忆错误", f"{type(exc).__name__}: {exc}")

    # ---------------- 脑子 → 嗓子 ----------------

    def _speak_loop(self) -> None:
        # 本轮是否已经出过声。**让路只对第二句之后生效** —— 否则别人在你
        # 刚要开口时插一句话,你就整轮不吭声了,体感是"叫它它不理我"。
        spoke_this_turn = False
        while not self._stop.is_set():
            try:
                item = self._sentences.get(timeout=0.2)
            except queue.Empty:
                continue

            sentence, display, epoch, gesture = item
            if epoch != self._epoch:
                # 过期货 —— 属于被打断/让路掐掉的那一轮,直接丢。
                # **不能拿它的结束标记去清 _speaking**:那清的是新一轮的状态,
                # 会让 AI 在自己的回合里听见自己说话。
                continue

            if sentence is None:
                spoke_this_turn = False
                self.osc.typing(False)
                self.osc.mouth(0.0)
                self._speaking.clear()
                continue

            # 只有动作、没有说话内容(LLM 单独吐了个 [挥手])——
            # 那就只发动作,不合成也不播放。注意 `""` 和 `None` 要分清:
            # None 是本轮结束标记(上面处理过了),空串才是"有动作没台词"。
            if not sentence:
                self._fire_gesture(gesture)
                continue

            # **句间让路**:念下一句之前先看一眼 —— 上一句念完的间隙里,
            # 有没有别人开口。有就把剩下的一起丢掉,把话权让出去。
            # 只在句与句之间判断,不在句子中间掐 —— 半句被切断听感很怪。
            if (spoke_this_turn and self.cfg.behavior.yield_to_others
                    and self._heard_other.is_set()):
                n = self._drain_sentences()
                self._epoch += 1            # 通知大脑:别再生成了
                self._heard_other.clear()
                self._barge_buf.clear()
                self.osc.mouth(0.0)
                self.osc.typing(False)
                self._speaking.clear()
                # 本轮结束标记会带着**旧**代号回来,会被上面的 epoch 检查丢掉 ——
                # 所以状态得在这里自己复位,不能指望那个标记。
                spoke_this_turn = False
                self.log("让路", f"听到别人说话,丢掉剩余 {n} 句")
                continue

            self.log("说", sentence)
            spoke_this_turn = True
            try:
                interrupted = self.tts.speak(
                    sentence,
                    on_level=self.osc.mouth,
                    # 字幕和**动作**都挂在这里:合成完成、马上要出声的那一刻。
                    # 挂到别处(比如收到句子时就发)会早于声音一个合成的时长。
                    # 动作挂同一个点 → 声音、字幕、身体三者天然同步。
                    on_ready=lambda d=display, g=gesture: (
                        self.osc.chatbox(d, force=True), self._fire_gesture(g)),
                    should_stop=self._barge.is_set,
                )
                if interrupted:
                    # 嗓门停了,脑子可能还在生成 —— 加代号通知它一起停。
                    self._epoch += 1
                    self._drain_sentences()
                    self._barge.clear()
                    self._barge_buf.clear()
                    self.osc.mouth(0.0)
                    # 同「让路」:结束标记是旧代号的,会被丢掉,状态自己复位。
                    spoke_this_turn = False
                    self.log("打断", "已停下")
            except Exception as exc:
                self.log("TTS错误", f"{type(exc).__name__}: {exc}")
                self.osc.mouth(0.0)
                self._speaking.clear()
