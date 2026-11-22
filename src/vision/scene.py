"""让小mo **看得见** —— 截图喂给视觉大模型,拿回一句场景描述,注入对话上下文。

## 为什么走"旁路"而不是换模型

`deepseek-chat` 是**纯文本**的,传图直接 400。DeepSeek 另有一个能吃图的
`deepseek-v4-flash-vision-exp`(2026-08 上线的实验模型,单图最多 384 tokens)。

**但不要整体换过去**,两个原因:
  1. **首句延迟会翻倍** —— 语音对话里每句都要传图,而我们的人设本来就 3300 字
  2. 人设、记忆、动作标记这套都是围绕 `deepseek-chat` 调的

所以**主链路一个字节都不动**,只在旁边开一路:

    主链路(不变):  deepseek-chat  ← 人设 + 记忆 + 「你现在看到的」这句话
                                          ↑ 注入
    旁路(异步):    deepseek-v4-flash-vision-exp
                        ↑  每 interval_s 跑一次,结果缓存
                    截图(WGC 已经在抓)

这样 **主链路的延迟完全不受影响**,视觉挂了说话照常。

## 三个刻意的设计

1. **异步、绝不阻塞对话。** 视觉调用要 1~3 秒,放在说话路径上就毁了。
   它自己一个线程,谁都不等它。
2. **画面没变就不重复调用。** 先做一次廉价的缩略图比对,变化不大就沿用
   上一条描述(只更新时间戳)。静止场景不该白烧 token。
3. **失败 = 没有视觉。** 抓不到图、网络挂了、模型下线了(它是 Exp)、
   返回垃圾 —— 一律当成"这条描述不存在",对话退化成今天之前的样子。
"""
from __future__ import annotations

import base64
import io
import threading
import time

import numpy as np

# 给视觉模型的指令。要点:
#   - **说人话、短** —— 这段话会被塞进小mo 的上下文,长了会挤掉别的
#   - **只要"能看到的事实"** —— 它看不到的(声音、想法)不许编
#   - **名字宁可不提也别猜** —— 截图上的名字是 VRChat 渲染的,可能读错
# ⚠️ **提示词必须短。** 这是个**带推理**的模型(reasoning_tokens 会先烧一截),
# 踩过的坑:第一版写了一长串约束(分点、强调、禁止事项),结果它**推理停不下来** ——
# max_tokens 给到 1500 仍然全部花在 reasoning 上、content 一个字都没有,
# 看着像"模型不支持图片",其实是提示词把它带进了死循环。
# 换成一句直白的指令,立刻正常(推理 136 / 输出 196)。
_VL_PROMPT = ("这是 VRChat 的画面截图。用一到两句话描述你看到的内容:"
              "有哪些人、环境是什么样的、正在发生什么。直接说,不要客套。"
              # 🔑 **认人一律以名牌为准,不看角色模型。**
              # VRChat 里角色模型经常加载不出来(默认灰模、或压根不渲染),
              # 但**名牌一直都在**。所以身份只能从名牌读,模型只是装饰。
              "注意:画面里那些悬浮的圆角名牌才是判断\"有谁\"的依据 ——"
              "角色模型经常加载不出来,但名牌总会显示,所以就算只看见名牌、"
              "看不见人,也要把名牌上的名字读出来。名字看不清就照实说看不清,别猜。"
              # 🔑 **聊天气泡 = 文字消息的身份通道。**
              # VRChat 把玩家打的字渲染成**挂在本人名牌上方**的黑色气泡 ——
              # 位置就绑定了"是谁说的",一次同时拿到「谁」和「说了什么」。
              # 这是语音拿不到的信息(身份在混音时就丢了,见 vrchat-speaker-id)。
              # ⚠️ **别只说"聊天气泡"。** 实测:模型看见了那个小黑框,却**不认为
              # 那是"气泡"** —— 同一个模型同一张图,一次说"黑色小框内写着 111"、
              # 另一次直接说"没有聊天气泡"。它认得字,只是不肯把它归到那个名字下。
              # 所以改成**要求把所有文字都读出来**,不依赖它对"气泡"的认知。
              "另外:画面里除了名牌,可能还有**别的文字** —— 比如挂在名牌上方的"
              "小黑色方框里玩家打的字。**把你看到的所有文字都一字不差地读出来**,"
              "并说明各自在画面的什么位置、附在什么东西上。"
              "读不出来的字就直说看不清,别猜、别漏。")


# 认出"做出小mo 的人"在画面里时,加在描述前面的一句**明确提示**。
# 为什么要程序来加、不让大模型自己认:视觉读名字**会读错**(实测 莫西干 → 夏西干/巽西干),
# 大模型拿到的就是错字,它没法自己把"夏西干9aa3"和"莫西干"联系起来。
# ⚠️ 这里写的名字是**说出口**要用的那个。名牌上其实是「莫西干 9aa3」,
# 但 `9aa3` 是 VRChat 的账号后缀,**念出来很怪** —— 只叫「莫西干」。
# (识别时仍然要靠 `9aa3` 去匹配,见 find_owner;识别用和念出来是两回事。)
_OWNER_NOTE = ("【画面里出现的这个人是「莫西干」—— "
               "做出你的那个人,也就是你的主人】\n")


def find_owner(text: str, aliases) -> str | None:
    """描述里有没有提到"主人"。命中返回命中的那个别名,否则 None。

    做**归一化后的包含匹配**(去空格标点、统一小写),这样
    `莫西干 9aa3` / `夏西干9aa3` 里的 `9aa3` 都能命中 —— **故意放宽**,
    因为漏认(主人站面前却不知道)比误认难堪得多,而且误认的代价只是
    小mo 对某个名字里带 9aa3 的人热情一点。
    """
    from src.vision.speaking import norm_name

    n = norm_name(text)
    if not n:
        return None
    for a in aliases or []:
        na = norm_name(a)
        if na and na in n:
            return a
    return None


# 句读 —— 出现在一块 OCR 文本里,就说明那是**聊天气泡**而不是名牌。
# 名牌长得像「莫西干9aa3°」,不会带逗号句号;气泡是句子,几乎一定带。
_BUBBLE_PUNCT = "，。！？；：、,.!?;:"


def owner_in_plates(plates, aliases) -> tuple[str, str] | None:
    """在 **OCR 名牌列表**里找主人。命中返回 `(别名, 名牌原文)`,否则 None。

    比 `find_owner` 准,因为两者的输入完全不同:

    | | 输入 | 闸门 |
    |---|---|---|
    | `find_owner` | **一大段自然语言描述** | 无 —— 名字出现在任意一句话里就算 |
    | `owner_in_plates` | **逐块 OCR 的名牌** | 长度 2~20、置信度 >0.5,且名字要落在一块牌子**内部** |

    🔴 描述的致命处在于它**混进了小mo 自己的字幕气泡**(`scene.py` 的提示词
    要求"把画面里所有文字一字不差读出来",而字幕正是挂在头顶的)。于是会**自激**:

        小mo 念「干嘛呀,莫西干。」→ 字幕挂头顶 → 下一轮描述里读到「莫西干」
        → 又判"主人在场" → 继续念「莫西干」

    名牌列表没这个回路 —— 它只读**名牌那一块**的文字。
    """
    from src.vision.speaking import norm_name

    for p in plates or ():
        raw = p.get("raw") or ""
        n = p.get("name") or ""         # 已经归一化过了,见 PlateTracker._loop
        if not n:
            continue
        # 🔴 **带句读的块 = 聊天气泡,不是名牌。**
        # 光换成名牌列表挡不住自激:气泡本身就是一块被 OCR 读到的文字,
        # 长度(8 字)和置信度都能过 `PlateTracker` 那两道闸。实测
        # `干嘛呀，莫西干。` 照样会被认成"主人在场"。
        # 而**名牌里不会有点号** —— 这是两者唯一稳定的结构差异。
        if any(ch in raw for ch in _BUBBLE_PUNCT):
            continue
        for a in aliases or ():
            na = norm_name(a)
            if na and na in n:
                return a, raw or n
    return None


class SceneDescriber:
    """后台定时"看"一眼画面,缓存一句描述。"""

    def __init__(self, cfg, *, llm_cfg=None, on_report=None):
        """
        `cfg`   = SceneConfig(见 src/config.py)
        `llm_cfg` = 复用的 LLMConfig,用来取 api_key_env / base_url(同一个 key)
        `on_report` = 可选回调 `(本次token, 累计token, 本次耗时s, 是否跳过)`,
                      给管线打日志用。**烧了多少 token 要看得见**,不然没法判断成本。
        """
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        self.on_report = on_report
        self.ok = False

        # token 账。跳过(画面没变)不计费,单独记次数。
        self.tokens = 0
        self.calls = 0
        self.skips = 0

        self._desc: str | None = None
        self._at: float = 0.0              # 最近一次**成功看过**的时刻
        self._pending_since: float = 0.0   # 最近一次**请求看**的时刻(wait_ready 用)
        self._prev_sig: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()      # 置位 = 立刻看一次(唤醒时用)
        self._thread: threading.Thread | None = None
        self._client = None

    # ---------------- 生命周期 ----------------

    def start(self) -> bool:
        """起后台线程。**返回 False 是正常情况**(没配、没 key、没窗口)。"""
        if not self.cfg.enabled:
            return False
        try:
            from openai import OpenAI

            key_env = (self.llm_cfg.api_key_env if self.llm_cfg else "DEEPSEEK_API_KEY")
            import os

            key = os.environ.get(key_env)
            if not key:
                return False
            self._client = OpenAI(
                api_key=key,
                base_url=(self.cfg.base_url
                          or (self.llm_cfg.base_url if self.llm_cfg else None)
                          or "https://api.deepseek.com"),
                timeout=self.cfg.timeout_s,
            )
        except Exception:
            return False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.ok = True
        return True

    def close(self) -> None:
        self.ok = False
        self._stop.set()

    # ---------------- 对外 ----------------

    def latest(self) -> str | None:
        """最近一次成功的描述。太旧或从没成功过就返回 None。

        ⚠️ 返回 None 是**常态**(没配视觉、刚启动还没看、抓图失败),
        调用方必须能接受 —— 那就等于"没有视觉",和小mo 以前一样。
        """
        with self._lock:
            if self._desc is None:
                return None
            if time.monotonic() - self._at > self.cfg.max_age_s:
                return None
            return self._desc

    def refresh_now(self) -> None:
        """请求立刻看一次(**不阻塞**,只是叫醒后台线程)。

        唤醒时调 —— 用户刚开口,画面大概就是他想让你看的。
        ⚠️ 会被 `min_gap_s` 限流:连续唤醒不会每次都真的调模型。
        """
        self._pending_since = time.monotonic()
        self._wake.set()

    # ---------------- 后台 ----------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            # **只在被叫醒时看。** 没有定时器 —— 定时看太烧 token 了:
            # 实测 20 秒一次 ≈ 13 万 tokens/小时,而 VRChat 画面**从不真正静止**
            # (待机动画、视角微晃),所以"画面没变就跳过"那个优化几乎从不生效。
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self._look()
            except Exception:
                pass          # 视觉绝不因异常把线程搞死

    def wait_ready(self, timeout: float) -> str | None:
        """**最多等 `timeout` 秒**等下一次"看"完成,然后返回描述。

        为什么需要它:看图要 ~3 秒,而唤醒判断发生在转写之后 —— 调用方
        如果直接问 `latest()`,拿到的多半是**上一次唤醒**留下的旧描述,
        或者是 None。等一小会儿能显著提高"这一轮就有画面"的概率。

        ⚠️ 但**不能等太久** —— 这是首句延迟,语音对话里等 3 秒是能听出来的。
        等不到就返回已有的(可能是 None),绝不硬撑。
        """
        if timeout > 0:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not self._stop.is_set():
                if self._looked_since(self._pending_since):
                    break
                time.sleep(0.05)
        return self.latest()

    def _looked_since(self, ts: float) -> bool:
        """`ts` 之后有没有完成过一次"看"。"""
        with self._lock:
            return self._at > ts

    def _look(self) -> None:
        # 限流:刚看过就别再看。人多嘴杂时可能连着被唤醒好几次,
        # 每次都调模型既烧钱又没意义(画面在几百毫秒里不会变)。
        # ⚠️ 这里要把 `_at` 顶上去,否则 `wait_ready()` 会傻等满超时。
        if self._at and time.monotonic() - self._at < self.cfg.min_gap_s:
            with self._lock:
                self._at = time.monotonic()
            self.skips += 1
            if self.on_report:
                try:
                    self.on_report(0, self.tokens, 0.0, True)
                except Exception:
                    pass
            return

        from src.vision import screen as S

        img = S.grab(self.cfg.window, client_only=True)
        if img is None:
            return

        # 和上次比一比 —— 画面没怎么变就别重复调 API 了
        small = self._signature(img)
        if self._prev_sig is not None and self._desc is not None:
            if float(np.abs(small - self._prev_sig).mean()) < self.cfg.change_min:
                with self._lock:
                    self._at = time.monotonic()      # 内容没变,但"看过"了
                self.skips += 1
                if self.on_report:
                    try:
                        self.on_report(0, self.tokens, 0.0, True)
                    except Exception:
                        pass
                return
        self._prev_sig = small

        t0 = time.monotonic()
        text, used = self._ask(img)
        if used:
            self.tokens += used
            self.calls += 1
        if self.on_report:
            try:
                self.on_report(used, self.tokens, time.monotonic() - t0, False)
            except Exception:
                pass
        if text:
            with self._lock:
                self._desc = text
                self._at = time.monotonic()

    @staticmethod
    def _signature(img: np.ndarray) -> np.ndarray:
        """把图缩成很小的一张灰度图当"指纹",用来判断画面变没变。"""
        h, w = img.shape[:2]
        step_y, step_x = max(1, h // 24), max(1, w // 32)
        small = img[::step_y, ::step_x].astype(np.float32)
        return small.mean(axis=2)

    def _ask(self, img: np.ndarray) -> tuple[str | None, int]:
        """把图发给视觉模型,拿回 (描述, 本次消耗的 token 数)。"""
        try:
            import cv2
        except Exception:
            cv2 = None

        h, w = img.shape[:2]
        # 缩到宽 cfg.max_width —— 传输快很多,而"看懂在干嘛"这点分辨率够了。
        # ⚠️ 用 JPEG 不用 PNG:同样内容体积小一个数量级。
        if cv2 is not None and w > self.cfg.max_width:
            scale = self.cfg.max_width / float(w)
            img = cv2.resize(img, (self.cfg.max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        if cv2 is not None:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if not ok:
                return None, 0
            raw = buf.tobytes()
        else:
            from PIL import Image

            bio = io.BytesIO()
            Image.fromarray(img).save(bio, format="JPEG", quality=self.cfg.jpeg_quality)
            raw = bio.getvalue()

        b64 = base64.b64encode(raw).decode("ascii")
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VL_PROMPT},
                        # ⚠️ 图**只能放在 user 消息里**,放 system/assistant 会 400
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=self.cfg.max_tokens,
                temperature=0.3,      # 描述要稳,不要发挥
                # 🔑 **关掉推理。** 这个模型默认会先"想"一大段,而且**极不稳定**:
                # 提示词稍微长一点它就会想个没完,把整个 max_tokens 吃光、
                # content 一个字都不剩(不报错,静悄悄地空)。
                # 实测同一张图同一个提示词,**3 次里 2 次全空**。
                #
                # 关掉之后:输出从 ~400 tokens 掉到 **55**,而且再没出现过空结果。
                # `reasoning_effort=low/minimal` 都不行 —— 仍然会烧 380~410。
                extra_body={"thinking": {"type": "disabled"}},
            )
            text = (resp.choices[0].message.content or "").strip()
            used = 0
            try:
                used = int(resp.usage.total_tokens or 0)
            except Exception:
                pass
            # 太长说明它在扯,截断 —— 这段要塞进对话上下文,不能占太多
            return (text[: self.cfg.max_chars] or None), used
        except Exception:
            return None, 0
