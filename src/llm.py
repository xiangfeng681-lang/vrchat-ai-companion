"""大脑 —— DeepSeek 流式生成,按句吐出。

用 OpenAI 兼容接口(deepseek 的 base_url),所以 SDK 是 `openai` 而不是
`anthropic`。选 DeepSeek 的主要原因不是便宜,是**国内直连延迟稳** ——
语音对话对延迟抖动比对延迟绝对值敏感得多。

这一层的核心不是「调用 API」,而是**按句切分**:
不等整段回复生成完再送 TTS,而是第一句一出现就丢过去合成。
这样首字延迟从「整段生成时间」降到「第一句生成时间」,体感差很多。

DeepSeek 侧的坑:
  * 系统提示要放进 messages 里当 {"role":"system"},没有顶层 system 参数
  * deepseek-reasoner 的流里会多出 reasoning_content 字段,要忽略掉
  * 流式最后一个 chunk 的 choices 可能是空的(带 usage 那种),要判空
  * 上下文缓存是自动的,命中自动打折,不需要手动标什么
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator

from openai import OpenAI

from .config import LLMConfig

# --------------------------------------------------------------------------
# 句子切分
# --------------------------------------------------------------------------

# 句末标点:见到就切
_SENTENCE_END = "。！？!?…\n"
# 兜底标点:太长了还没句末标点,就退而求其次切在子句边界
_CLAUSE_END = "，,、；;：:"
# 缓冲到这么多字符还没句末标点 → 找最近的逗号切
_FORCE_SPLIT_AT = 60
# 到这个长度还是没有任何标点 → 硬切,再等下去延迟就失控了
_HARD_SPLIT_AT = 100
# 句子的最小"实字"数。**少于这个的碎片先攒着,和后面并成一句。**
#
# 为什么需要:每个句子会被**单独送去 TTS 合成、单独播放**,中间还隔着一次
# 网络往返。所以「那…」这种碎片会变成:一声孤零零的"那"→ 明显停顿 → 后半句。
# 实测小mo 很爱用「嗯…」「那…」开头,不合并的话听起来一顿一顿的。
_MIN_SENTENCE_CHARS = 6


def _content_len(s: str) -> int:
    """"实字"数 —— 标点和空白不算,免得「…」「。」这种也算一句话。"""
    return sum(1 for c in s if c.isalnum() or "一" <= c <= "鿿")


def split_sentences(buf: str) -> tuple[list[str], str]:
    """从缓冲里切出所有完整句子,返回 (句子列表, 剩余缓冲)。

    太短的碎片**不单独吐出**,而是攒进 pending 和后面的并成一句。
    没凑够的部分会跟着剩余缓冲一起还回去,下一轮再接着攒。
    """
    out: list[str] = []
    rest = buf
    pending = ""
    while True:
        cut = _find_cut(rest)
        if cut is None:
            break
        piece, rest = rest[:cut].strip(), rest[cut:]
        if not piece:
            continue
        pending = f"{pending}{piece}" if pending else piece
        if _content_len(pending) >= _MIN_SENTENCE_CHARS:
            out.append(pending)
            pending = ""
    # pending 里可能还攒着没凑够的碎片 —— 拼回缓冲,别丢
    return out, f"{pending}{rest}" if pending else rest


def _find_cut(buf: str) -> int | None:
    """决定在哪里切一刀。返回切点(不含),None 表示还不到切的时候。

    切得越碎延迟越低,但每一句 TTS 都要付一次固定开销(edge-tts 约 1.8s),
    而且半截话念出来语气也怪。所以策略是:
      1. 见到句末标点立刻切 —— 这是最自然的边界
      2. 拖到 60 字还没句末标点,退而在最近的逗号处切
      3. 拖到 100 字连逗号都没有,硬切,别让延迟失控
    """
    for i, ch in enumerate(buf):
        if ch in _SENTENCE_END:
            return i + 1

    if len(buf) >= _HARD_SPLIT_AT:
        return _HARD_SPLIT_AT

    if len(buf) >= _FORCE_SPLIT_AT:
        # 从后往前找最近的逗号,尽量切在语义边界上
        for i in range(len(buf) - 1, _FORCE_SPLIT_AT // 2, -1):
            if buf[i] in _CLAUSE_END:
                return i + 1

    return None


# --------------------------------------------------------------------------
# 运行约束(附加在人设后面,不要写进 config.yaml)
# --------------------------------------------------------------------------

# 注入临时记忆用的标记。带这个键的消息回答完就摘掉,不进历史。
_SCENE_TAG = "_scene_note"

# 视觉模型"看"出来的场景描述。**由程序注入,不是用户说的话**,必须标注清楚,
# 否则模型会以为用户在描述自己的画面。
# ⚠️ 三条约束都是踩点:**只信这条、别脑补、别提错名字**。
_SCENE_TEMPLATE = """【你现在看到的画面 —— 由摄像头自动分析,不是用户说的话】
{text}

🔑 **如果描述里提到某人头顶的打字框/小方框里写着什么,那就是他"说的话"。**
游戏里打字和语音是两种说话方式,**对听者来说没有区别** ——
所以被问"他说什么了"时,**打字框里的内容就是答案**,别因为"那是打字不是说话"
就去声音里找、然后说"没听到"。

使用要求:
- 这是**几秒前**的画面快照,可能已经过时。和用户当下说的冲突时,**以用户说的为准**。
- 只在**自然相关**时提,别每句都报一遍场景,那样很烦。
- 描述里没有的东西**就是你看不到的**,别顺着"你能看到"去脑补画面里有什么。
- 画面里的名字是文字识别读出来的,**可能读错** —— 不确定就别点名,
  宁可说"你旁边那个人"。"""


_MEM_TAG = "_memory_note"

_MEMORY_TEMPLATE = """【你记得的往事 —— 由记忆系统自动检索,不是用户这轮说的话】
{items}

🔴 **最重要的一条:先回答用户这一轮问的那句话。**
往事是**背景**,不是**话题** —— 用户问 A,你不能答成 B。

判断方法:把往事和用户这轮的话放在一起看。
**有直接关系**才提;**没有直接关系就整轮都别提**,当它不存在。
往事再有趣,也不能顶替对当前这句话的回应。

⚠️ 这条是补上去的,因为实测踩过:用户问「其他人跟你说了什么」,
检索到的往事里有一条旧事聊到"带我去逛世界",结果回出来一句
「谁特意陪你逛了?」—— **完全没回答用户的问题**,顺着往事跑偏了。

其余要求:
- 只在**自然相关**的时候提,别每句都硬扯。
- 别生硬复述("根据我的记忆…"),像人一样随口带出来就行。
- 这些片段可能记错或过时,和用户当下说的冲突时,**以当下为准**。"""


_VOICE_RULES = """
【运行约束 —— 这段由程序自动追加,不要当作角色设定】
- 这是实时语音对话,你的回复会被 TTS 念出来。默认 1~2 句话;对方明确要求详细说明时才展开。
- 用口语化中文,像在语音频道里聊天。
- 不要用 Markdown:不要 **加粗**、# 标题、- 列表、表格、代码块、链接。念出来很奇怪。
- 不要在回复里包含内部或系统标签,也不要输出任何括号里的动作描写。
"""


class ChatSession:
    """多轮对话 + 流式按句输出。"""

    def __init__(self, cfg: LLMConfig, persona: str, *, client=None):
        self.cfg = cfg
        self.client = client or OpenAI(
            api_key=self._resolve_key(cfg),
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
        )
        self.messages: list[dict] = [
            {"role": "system", "content": persona.strip() + "\n" + _VOICE_RULES}
        ]
        self.last_finish_reason: str | None = None
        self._abort = threading.Event()   # 「闭嘴」打断用,见 abort()

    @staticmethod
    def _resolve_key(cfg: LLMConfig) -> str:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(
                f"环境变量 {cfg.api_key_env} 未设置。\n"
                f"去 platform.deepseek.com 拿一个 key,然后:\n"
                f"    setx {cfg.api_key_env} sk-xxxxxxxx\n"
                f"设完要**重开终端**才生效。"
            )
        return key

    # ---------------- 对话 ----------------

    def abort(self) -> None:
        """叫停正在进行的流式生成(「闭嘴」打断用)。

        只置个标志 —— 真正的退出发生在 `reply()` 的循环里,它每收到一个
        chunk 检查一次。这样不必去动 openai SDK 的连接对象。
        """
        self._abort.set()

    def reply(self, user_text: str, memories: list[str] | None = None,
              scene: str | None = None) -> Iterator[str]:
        """流式产出回复的**一句一句**文本。

        `memories` 是向量记忆捞出来的相关往事,作为**临时** system 消息插在
        本轮用户消息前面 —— 回答完就摘掉,不留在历史里。留着的话每轮都会
        累积,既撑爆上下文,又会让它反复念叨同样的事。

        `scene` 是视觉模型看出来的一句画面描述(见 vision/scene.py)。同样是
        **临时**注入、答完就摘。`None` = 没有视觉信息,行为和以前完全一致。
        """
        self.messages.append({"role": "user", "content": user_text})
        if scene:
            self.messages.insert(
                len(self.messages) - 1,
                {
                    "role": "system",
                    "content": _SCENE_TEMPLATE.format(text=scene),
                    _SCENE_TAG: True,
                },
            )
        if memories:
            self.messages.insert(
                len(self.messages) - 1,
                {
                    "role": "system",
                    "content": _MEMORY_TEMPLATE.format(
                        items="\n".join(f"- {m}" for m in memories)
                    ),
                    _MEM_TAG: True,
                },
            )
        self._trim_history()

        self._abort.clear()            # 新的一轮,清掉上轮的叫停标志
        buf = ""
        full = ""
        finish: str | None = None

        # 🔴 **必须关掉"思考模式"。**
        #
        # `qwen3.8-flash` 是**推理模型** —— 它会先吐一大段 `reasoning_content`
        # 再出正文字。而下面是流式的,那段思考**一个字都念不出来**,
        # 纯粹是白等。实测(同一句、同一个人设):
        #
        #     开着思考   首字 +7.34s   思考 583 字
        #     关掉思考   首字 +1.09s   思考   0 字      ← 快 6.7 倍
        #
        # 更糟的是它**随人设长度增长**:人设 8100 字时首句 ~5s,
        # 加到 9166 字后实测出现过 **+71s** —— 语音对话里这等于"它不理你"。
        # 记忆里眼睛那个模型(`deepseek-v4-flash-vision-exp`)栽的是同一个坑。
        _extra = {}
        if not getattr(self.cfg, "enable_thinking", False):
            _extra["extra_body"] = {"enable_thinking": False}

        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=self.messages,
            stream=True,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            **_extra,
            # 不传 top_p 的话 API 默认是 1.0 = 不设限,长尾里的胡话会漏出来。
            # 0.9 保留表达自由但把候选收在累计概率 90% 以内。
            top_p=self.cfg.top_p,
            # 防复读。文章里叫 repetition_penalty,OpenAI 兼容接口里叫
            # frequency_penalty —— 小mo 偶尔会连着几句用同一个开场白。
            frequency_penalty=self.cfg.frequency_penalty,
        )

        for chunk in stream:
            if self._abort.is_set():
                stream.close()         # 主动断开,别让服务端把剩下的白算完
                full = full or "（被打断）"
                break
            if not chunk.choices:      # 末尾带 usage 的空 chunk
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason

            delta = choice.delta.content if choice.delta else None
            if not delta:              # reasoning_content 之类,直接跳过
                continue

            full += delta
            buf += delta
            sentences, buf = split_sentences(buf)
            for s in sentences:
                yield s

        if buf.strip():
            yield buf.strip()

        self.last_finish_reason = finish
        # 摘掉本轮临时注入的**记忆和画面** —— 它们只服务于这一轮。
        # 留着的话每轮累积,上下文很快被撑爆;而且画面会一直停在那一帧上,
        # 让它反复描述同一张旧图。
        self.messages = [m for m in self.messages
                         if not m.get(_MEM_TAG) and not m.get(_SCENE_TAG)]
        self.messages.append({"role": "assistant", "content": full.strip() or "..."})

    def _trim_history(self) -> None:
        """只保留最近 N 轮。messages[0] 是 system,永远留着。"""
        limit = max(2, self.cfg.history_turns * 2)
        body = self.messages[1:]
        if len(body) <= limit:
            return
        body = body[-limit:]
        while body and body[0]["role"] != "user":
            body.pop(0)
        self.messages = self.messages[:1] + body

    def reset(self) -> None:
        self.messages = self.messages[:1]
