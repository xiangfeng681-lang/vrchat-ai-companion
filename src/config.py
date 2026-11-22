"""配置加载。

config.yaml -> 一组 dataclass。缺项一律走默认值,所以删掉任何一段都能跑。
注意:Windows 中文环境下 open() 默认是 GBK,这里必须显式指定 utf-8。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


# --------------------------------------------------------------------------
# 各段配置
# --------------------------------------------------------------------------

@dataclass
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    max_tokens: int = 512
    # ⚠️ DeepSeek 的温度尺度和别的模型**不一样**:
    #   官方建议 代码 0.0 / 数据分析 1.0 / 通用对话 1.3 / 创意写作 1.5。
    #   换算成"通用尺度"大约是打个对折,所以 1.1 相当于别家的 0.55~0.7。
    temperature: float = 1.1
    # 核采样。**之前完全没设,走的是 API 默认 1.0 = 不设限**。
    # 0.9 是把候选词限制在累计概率 90% 以内 —— 保留表达自由,但挡住长尾里的
    # 胡话。出问题时的调法:胡言乱语 ↓0.15,重复啰嗦 ↑0.05。
    top_p: float = 0.9
    # 频率惩罚:按"这个词已经出现过几次"扣分,专治复读。
    # ⚠️ 别调高 —— 它不认识"专有名词"和"口头禅",调猛了会让小mo 连自己的
    # 名字都不愿意重复说。0.2~0.4 是"偶尔复读"的合适剂量。
    frequency_penalty: float = 0.3
    # 🔴 **推理模型的"思考"必须关掉。**
    #
    # `qwen3.8-flash` 是推理模型:它先吐一大段 `reasoning_content` 才出正文。
    # 而管线是**流式按句念**的,那段思考一个字都念不出来 —— 纯粹白等,
    # 而且**随人设长度增长**。实测(同一句、同一人设):
    #     开着  首字 +7.34s(人设 9166 字时实测出现过 +71s!)
    #     关掉  首字 +1.09s      ← 快 6.7 倍
    # 语音对话里 71 秒等于"它不理你"。只在查"它为什么答得怪"时才值得打开。
    enable_thinking: bool = False

    history_turns: int = 12
    timeout_s: float = 30.0


@dataclass
class SceneConfig:
    """让小mo **看得见** —— 截图喂给视觉模型,拿回一句场景描述。

    走**旁路**:主链路还是 `deepseek-chat`(纯文本),这里单独调一个能吃图的模型,
    结果当成一句话注入对话。**所以主链路的延迟完全不受影响**(验证见 scene.py)。
    """
    enabled: bool = True

    # ⚠️ **别填 `deepseek-chat`** —— 它不接受图片,会直接 400。
    #    能吃图的是这个实验模型(2026-08 上线,单图最多 384 tokens)。
    #    留空 = 关掉视觉。
    model: str = "deepseek-v4-flash-vision-exp"

    # 留空就复用 llm 段的 base_url 和 api_key_env(同一家的 key)
    base_url: str = ""

    window: str = "VRChat"

    # 🎯 **只在被唤醒时"看"一眼。**
    #
    # 为什么不定时看:实测 20 秒一次 ≈ **13 万 tokens/小时**,而且那套
    # "画面没变就跳过"的优化**几乎从不生效** —— VRChat 画面从来不真正静止
    # (待机动画、视角微晃),每次都判定为"变了"。
    #
    # ⚠️ 代价:**看图要 ~3 秒,而首句 2 秒就出去了**。所以唤醒那一轮基本
    #    用不上画面,得等下一轮。下面的 `wait_s` 是拿延迟换命中率的旋钮。
    min_gap_s: float = 12.0

    # 回答之前**最多**等多久等这次"看"完成(秒)。0 = 一秒都不等。
    # 实测看图 1.4~4.3 秒,所以等 1~2 秒只能捞到"已经快好的"那次;
    # 想每轮都有画面就得给到 3.5 以上,代价是首句明显变慢。
    wait_s: float = 1.2

    # 描述超过这么久没用上就作废 —— **宁可没有视觉,也别让小mo 拿着
    # 几分钟前的画面当真**(它会说"你旁边那个人",其实人早走了)。
    # 只在唤醒时看的话,这个值别太大。
    max_age_s: float = 120.0

    # 和上一帧的平均像素差小于这个值就认为"没变",不重复调模型(省 token)。
    # ⚠️ 实测 3.0 太严(VRChat 永远在微动,基本从没触发过跳过),
    #    真要省 token 就往上调;8~10 才会开始生效。
    change_min: float = 3.0

    max_width: int = 768      # 缩到这个宽度再传,够看懂,传输快(实测 25KB/张)
    jpeg_quality: int = 80    # 用 JPEG 不用 PNG,体积小一个数量级

    # **推理已关**(见 scene.py 的 `extra_body`),所以这里不用再给推理留余量。
    # 实测关掉推理后一次只用 **55** tokens;600 是宽松的保险值。
    # (关推理之前这个值很关键 —— 给 200 会被推理吃光、给 1500 也一样是空。)
    max_tokens: int = 600
    max_chars: int = 200      # 描述截断长度 —— 要塞进上下文,不能占太多
    timeout_s: float = 30.0   # 带推理,比纯文本慢;实测单次 3~9 秒


@dataclass
class OscConfig:
    host: str = "127.0.0.1"
    port: int = 9000


@dataclass
class ChatboxConfig:
    enabled: bool = True
    send_sound: bool = False
    min_interval_s: float = 1.5
    max_chars: int = 144


@dataclass
class AudioConfig:
    capture_device: Any = "loopback"   # "loopback" / "loopback:名字片段" / 名字片段 / 索引
    output_device: Any = None          # AI 的嗓子:VB-CABLE 的 "CABLE Input"
    block_ms: int = 30
    # 采集采样率。回环设备通常强制 48000(Windows 共享模式),设了也会被忽略,
    # 实际用的是设备原生率 —— 代码会自动按真实率重采样给 Whisper。
    preferred_rate: int = 48000


@dataclass
class VadConfig:
    energy_threshold: float = 0.012
    silence_ms: int = 700
    min_speech_ms: int = 300
    max_utterance_ms: int = 15000
    preroll_ms: int = 300


@dataclass
class AsrConfig:
    # ---- 后端选择 ----
    # `local` = 本地 faster-whisper;**`cloud` = 云端**(阿里云百炼)。
    #
    # 🔑 换云端能一次解决三个问题:**省 ~2G 显存**(本地 Whisper 常驻显存)、
    #    不再跟 VRChat 抢 GPU、打断检测那条每 0.8 秒跑一次的通道也不占 CPU 了。
    #
    # ⚠️ 代价:**断网就聋**。本地 Whisper 是唯一不怕断网的方案。
    backend: str = "local"

    # 云端模型。实测 `qwen3.5-omni-flash`:0.4 秒、自带标点、准确率比本地 large-v3 还好。
    #
    # ⚠️ **别改成 `fun-asr-flash` 之类** —— 百炼的 OpenAI 兼容接口**没有**
    #    `/audio/transcriptions`(实测 404),音频只能走 chat 接口的 `input_audio`,
    #    而那条路目前只有全模态(omni)模型认。
    cloud_model: str = "qwen3.5-omni-flash"
    cloud_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    cloud_api_key_env: str = "DASHSCOPE_API_KEY"
    cloud_timeout_s: float = 20.0
    # 云端失败(网络挂了/额度用完)时要不要退回本地模型。
    # ⚠️ 设 True 的话本地 Whisper 会被**懒加载**,也就是失败一次之后显存又占回去了。
    #    想彻底省显存就保持 False。
    cloud_fallback_local: bool = False

    # 🔑 **云端没有 hotwords,这是唯一能拿回"热词偏置"的地方。**
    #
    # 本地 Whisper 有 `hotwords` 参数(记忆:"**热词偏置** —— 解决'叫了没反应'
    # 的正解",当时把被吞成「那」的「小莫」救了回来)。但**云端走 chat 接口,
    # 没有那个参数** —— 换过来之后偏置丢了,「叫了没反应」又回来,
    # **别人的嗓音更明显**(自己的声音它听惯了)。
    #
    # 只能在提示词里"预告"可能出现的词。⚠️ **必须短** —— 这个模型偏"想",
    # 提示词越长越容易跑偏(眼睛那个模型栽过:token 全烧在思考上、正文全空)。
    cloud_hint: str = (
        "提示:说话人可能会叫「小莫」这个名字(也可能写作小茉/小墨/小默),"
        "听到这个音就照实写出来,不要改写成别的词。"
    )

    model: str = "small"
    language: str = "zh"
    compute_type: str = "int8"
    device: str = "auto"

    # 🔑 **热词偏置** —— 直接在**解码时**告诉 Whisper"这些词更可能出现"。
    #
    # 为什么需要:实测用户喊「小莫你看我说你还在吗」,被转写成
    # 「**那**你看我说你还在吗」—— 「小莫」整个被吞成了「那」,
    # 唤醒词匹配不上,表现为"**叫了没反应**"。
    #
    # 加同音字表治不了这个(**它不是同音字错,是整词被替换**),
    # 但 `hotwords` 是在束搜索打分时加偏置,正好能救。
    #
    # ⚠️ 比改 `initial_prompt` 安全得多 —— 那个是"把提示词当上文续写",
    #    喂静音时会诱发幻觉(所以要配 `_looks_like_prompt_echo` 那道闸);
    #    `hotwords` 只是词表偏置,不参与续写。
    #
    # 空格分隔。填关键词就够了,**别塞太多** —— 会把别的词也往这些读音上拽。
    hotwords: str = "小莫 小茉 小墨 小mo 莫西干"

    # 转写纠错表 —— Whisper 把**专有名词**听错时按表替换。
    #
    # 实测:**「莫西干」会被听成「墨西哥人」**(mò xī gān / mò xī gē rén 太像)。
    # 这个错比一般的听错麻烦 —— 它会**顺着记忆存下去**,以后检索出来的
    # 都变成"墨西哥人做的",错得很持久。
    #
    # ⚠️ 这是**全局字符串替换**,写的时候要挑"正常对话里几乎不会出现"的错法。
    #    「墨西哥人」在这个场景里基本不会正常出现,所以安全。
    # ⚠️ **不要改成动 `initial_prompt`** —— 那个提示词是幻觉的主要来源,
    #    改它就要重跑 `_looks_like_prompt_echo` 的阈值验证,风险大得多。
    fix_words: dict[str, str] = field(default_factory=dict)
    # Whisper 对中文会随机吐繁体(训练数据繁简混杂)。给一段简体提示词
    # 能把它稳定在简体上 —— 这是标准做法,比装 opencc 做转换省事。
    # ⚠️ **只在 language 写死时才会送进去**(见 asr.transcribe):它是纯中文的,
    #    自动识别语种时拿它做偏置会把英/日的结果往中文拽。所以 language=auto
    #    的代价之一,就是繁体回潮(实测 7 句里 4 句繁体)—— 匹配用的副本已经
    #    做了繁简归一(见 asr.SIMPLIFY),不影响唤醒词和停止词。
    initial_prompt: str = "以下是普通话的句子,请用简体中文转写。"

    # 一段被 Whisper 判为「不是语音」的概率超过这个值,就整段丢掉。
    #
    # 为什么是 0.8 而不是更保守的 0.6:实测**纯噪声/静音会给 0.83~0.86**
    # (Whisper 自己知道没人说话),所以 0.8 刚好卡在噪音下限上面一点。
    # 定在 0.6 会在 0.6~0.83 之间白扔真话 —— 实测抓到的正是这种情况:
    # 「空结果[无人声 0.68]」连着两次,那是**真人说话被误杀**。
    #
    # 当初压到 0.6,是因为这道闸是那时唯一有效的防幻觉手段。现在下游还有
    # 提示词幻觉、媒体残留黑名单、重复检测三道,第一道没必要再卡那么死。
    # 调法:老是漏真话就往上加(0.85),幻觉变多就往下减。
    no_speech_max: float = 0.8
    # 首次下模型走这个镜像。**默认就是镜像** —— 这台机器连不上 huggingface.co,
    # 留空会 ConnectTimeout。想用官方源就设成 ""。
    hf_endpoint: str = "https://hf-mirror.com"


@dataclass
class TtsConfig:
    # edge = 微软在线合成(免费、零显存、但固定 ~1.85s 开销且不能克隆音色)
    # gpt_sovits = 本地克隆(吃 2~3GB 显存、RTF≈0.48、能克隆任意音色、不怕断网)
    # cosyvoice = 阿里云百炼**云端克隆**(零显存、能克隆、⚠️ 断网就哑)
    backend: str = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+15%"
    volume: str = "+0%"

    # ---------- 以下仅 backend=gpt_sovits 时生效 ----------
    api_url: str = "http://127.0.0.1:9880"
    ref_audio: str = ""          # 参考音频路径(相对 GPT-SoVITS 根目录)
    prompt_text: str = ""        # 参考音频里说的那句话,必须准,错了音色会跑偏
    prompt_lang: str = "zh"
    speed_factor: float = 1.0
    temperature: float = 1.0     # 调低更稳,调高更飘
    top_k: int = 15
    top_p: float = 1.0
    timeout_s: float = 120.0

    # ---------- 以下仅 backend=cosyvoice 时生效 ----------
    # 阿里云百炼的云端声音复刻。**迁到 GTX1060(6G) 就是为了它** ——
    # 把 GPT-SoVITS 那 2~3GB 显存让给 VRChat。
    #
    # 🔴 合成模型**必须和注册音色时的 target_model 逐字一致**,否则合成直接失败
    #    (不是"音色差一点",是报错)。注册见 tools/enroll_dashscope_voice.py。
    cloud_model: str = "cosyvoice-v2"

    # 注册拿到的 voice_id。**留空 = 不碰云端,直接走 edge**(免得每句白等一次)。
    # 形如 cosyvoice-v2-<prefix>-<32位hex>。
    cloud_voice_id: str = ""

    # key 从环境变量取,和 asr / llm 用**同一个**(一个百炼账号一把 key)。
    cloud_api_key_env: str = "DASHSCOPE_API_KEY"

    # 云端超时。⚠️ 别抄上面 gpt_sovits 那个 120s —— 本地服务慢是正常的,
    # 云端 30s 没回就是出事了,拖长只会让这句话卡住(而后面还排着队)。
    cloud_timeout_s: float = 30.0

    # 云端失败**不兜底** —— 直接抛异常,这句话跳过不说。
    #
    # 🔑 为什么不退到 edge(这曾是设计默认):**实测 edge-tts 在这台机器上已经
    # 不可用** —— 微软 speech.platform.bing.com 持续返回 403(连试 3 次全失败,
    # edge-tts 7.2.8 已是最新版、系统时间也正常,大概率是区域限制)。
    # 也就是说"退到 edge" = 白等一次握手再失败,没有任何收益。
    #
    # 而抛异常是**安全**的:pipeline 会记 `TTS错误`、嘴巴归零、清掉 speaking 状态,
    # 这一句直接跳过。**不会出现"有字幕没声音"的哑剧** —— 因为 on_ready
    # (发字幕、做动作)挂在 synthesize **之后**,抛异常根本到不了那一步。
    #
    # 连续失败几次就熔断(本进程内一段时间不再试云端):断网时每次都要白等
    # cloud_timeout_s,LLM 一口气吐 5 句就要卡 150 秒。熔断后立刻失败,不等。
    cloud_breaker_n: int = 3

    # 合成前做文本替换。中英混排会让 G2P 在语言边界插停顿 ——
    # 实测「小mo」比「小茉」长 0.68s。只替换送去合成的文本,
    # 字幕和对话记录里仍然是原样,用户看不出区别。
    name_fix: dict = field(default_factory=dict)


@dataclass
class WakeConfig:
    """唤醒词 —— 只在被叫到名字时才回应。

    解决的是**多人场合乱插嘴**:没有唤醒词时,任何人的任何一句话都会被
    当成对它说的,于是别人聊天它就抢话。加上唤醒词后,它只在听到
    「小莫」之类的称呼时才应答。

    实现方式是**转录后过滤**,不是专门的唤醒词引擎(openWakeWord 之类)——
    因为管线本来就要把每句话都转录一遍,顺手看一眼有没有叫名字,零额外成本。
    代价是没有"提前拦截",省不了 ASR 那 0.3 秒,但那个开销可以忽略。
    """

    enabled: bool = False
    # ASR 会把「小莫」听成「小茉/小墨/小mo」等,所以给一组变体一起匹配。
    # 想更省事就在名字里加个生僻字,但太生僻 ASR 又容易听错,是个平衡。
    words: list = field(default_factory=lambda: ["小莫", "小茉", "小墨", "小mo"])
    # 唤醒后 N 秒内不用再叫名字,可以连着说几句 —— 否则每句都要"小莫…"很累。
    followup_s: float = 15.0
    # 比这还短的碎片直接丢,减少误触发和"嗯""啊"这类噪声。
    min_chars: int = 2
    # 叫了名字但后面没别的话时,是否也回应(比如只喊了一声"小莫?")。
    reply_to_name_only: bool = True

    # 「闭嘴」类停止词 —— 让它正在说的话立刻停下。
    # 用**短关键词**而不是长句:ASR 不一定把长句听全,但"闭嘴"两个字很难漏。
    # 留空 = 关掉打断功能,回到以前"它说话时你插不上嘴"的行为。
    stop_words: list = field(default_factory=lambda: ["闭嘴", "别说了", "停下", "安静"])
    # AI 说话期间每隔多久检查一次停止词(秒)。越短打断越快,但 ASR 跑得越勤。
    # 0.8 秒 ≈ 每次 Whisper 约 0.3s(GPU),占用可接受。
    stop_check_s: float = 0.8


@dataclass
class MemoryConfig:
    """向量记忆 —— 让小mo 记得住 12 轮上下文之外的事,以及跨重启的事。

    LLM 上下文只有 `history_turns` 轮,超了就丢、重启就忘。这里把每轮对话
    存进向量库,每次回答前按语义相关度捞几条塞回上下文。
    """

    enabled: bool = False
    db_path: str = "memory.db"      # 相对项目根目录
    top_k: int = 3                  # 每轮最多注入几条
    # 低于这个余弦相似度就当"不相关",不注入。
    # **宁可不给也别硬塞** —— 塞错记忆比没有记忆更糟,它会让 AI 无中生有。
    min_similarity: float = 0.45
    min_chars: int = 4              # 太短的("嗯""好的")不值得存
    max_entries: int = 5000         # 库上限,超了删最旧的
    max_tokens: int = 64            # 嵌入时截断长度(记忆句子都不长)
    cpu_threads: int = 2            # 嵌入走 CPU,别跟 Whisper 抢满核


@dataclass
class VoiceprintConfig:
    """声纹 —— 判断「这句话是不是主人说的」。

    用途是**权限分级**:主人能下移动指令(跟随/前进),别人只能触发动作和跳舞。
    和视觉的"发光描边"是互补的 —— 那个知道画面里谁在说,这个**背对着也认得出**。

    ⚠️ **认不出 = 当非主人**,是安全的那一侧:最坏是主人要重说一遍,
    不会出现"陌生人把小mo 带走"。
    """

    enabled: bool = False
    model: str = "models/speaker/campplus_zh.onnx"   # 相对项目根
    profile: str = "models/speaker/owner.npy"        # 注册出来的参考向量
    # 阈值来自 2026-09-18 的 A/B:主人最低 0.479 / 其他人最高 0.360,
    # 中间取任意值都行,**0.42 是那个空档的中点**。
    # ⚠️ 负样本只有一段(约 2 个人),人多起来要重验。
    threshold: float = 0.42


@dataclass
class BehaviorConfig:
    mute_input_while_speaking: bool = True
    chatbox_echo_user: bool = False
    # 回声守卫:AI 说完话后,它自己的声音会从**对方那边绕一圈回来**
    # (AI → VRChat → 对方手机喇叭 → 对方手机麦克风 → 传回 PC → 播到耳机
    #  → 我们的耳朵)。这条环路物理上消不掉,但我们**知道自己刚说了什么**,
    # 所以把新听到的句子和最近说过的比一比,像的就当没听见。
    # 设 0 关闭。常见的回声往返在 1~3 秒,给 10 秒余量足够。
    echo_guard_s: float = 10.0
    # 相似度超过这个比例就判定为"自己刚说的"。
    # 太高→回声漏过去(它开始自言自語);太低→把用户的正常重复也误杀。
    echo_guard_ratio: float = 0.72

    # 一轮最多念几句。0 = 不限。
    # 人设里写了"只说一到两句",但提示词只是**劝** —— 拦不住的时候得有人动手。
    # 超出上限的句子直接丢,并掐断还在流式的 LLM 请求。
    max_sentences: int = 2

    # 句间让路:说话期间听到**别人**在说话,就把剩下没念的句子丢掉,让出话权。
    # 只在句与句之间判断,不在句子中间掐 —— 半句被切断听感很怪。
    yield_to_others: bool = True

    # 动作联动:LLM 在回复里用 [挥手] 这样的标记点动作,
    # 管线把它摘出来、**挂在"这一句即将出声"那一刻**发给动作播放器
    # (tools/pose_server.py)。这样动作和声音天然同步。
    # 设 0 关掉联动(就不会去连那个端口)。
    gesture_port: int = 39600

    # ---- 转向说话的人 ----
    # 听到有人在说话时,转头+转身**看向他**。
    #
    # 怎么知道"他在哪边":**左右靠立体声**(VRChat 的玩家语音是空间音频渲染的,
    # 人在左边声音就偏左,见 audio/capture.py::direction),**是谁靠 OCR 读名牌**。
    #
    # ⚠️ **只能分左右,分不了前后** —— 两个声道下"正前方"和"正后方"的左右
    #    能量完全相同,这是信息论限制。人在背后时会退化成"看着正前方"。
    look_at_speaker: bool = True

    # 横向偏移 → 偏航角 的换算用的视场角(度)。
    # 名牌在画面最边上(cx=0 或 1)时转多少度。VRChat 桌面视角横向大约 70°。
    look_fov: float = 70.0

    # 音频方向小于这个值就当"人在正前方",**不转** ——
    # 宁可不动,也别在人家站你正前面时乱扭头。
    look_deadzone: float = 0.03

    # 视野里**看不见**说话人时,退回只靠声音估角度(度 / 每单位平衡值)。
    #
    # 实测人在侧方约 45° 时平衡值 ≈ 0.05~0.10,所以 300 对应 15~30°。
    # ⚠️ **精度远不如视觉**(还受距离影响:离得越近平衡越极端),
    #    但它补的是视觉的**死角** —— 人在你侧后方说话,名牌在画面外,
    #    而那恰恰是最该转身的情况。总比一动不动强。
    # 设 0 = 关掉这条退路(宁可不动也不乱转)。
    look_audio_gain: float = 300.0

    # 发出转向指令后,保持这么多秒再**自动回正**(pose_server 那边执行)。
    # ⚠️ **必须和 `tools/pose_server.py` 的 `LOOK_HOLD_S` 保持一致** ——
    # 管线靠它判断"那边是不是已经自己回正了"。不一致会累积出错误的角度。
    #
    # 别设太短:实测 6 秒时,说着说着它就转回正面了,看着像心不在焉。
    # 30 秒大致对应"一次对话的间隔"。
    look_hold_s: float = 30.0

    # ---- 「招牌反击」硬闸 ----
    # 🔴 **提示词是劝,拦不住的时候得有人动手** —— 和 `max_sentences` 一个道理。
    # 实测:光靠人设写"只在被挑衅之后说",它还是会在没被逗的时候冒出来。
    #
    # 规则:**只有当用户这句话里出现了挑衅词,才允许回复里出现招牌句**;
    # 没出现就把招牌句从回复里剥掉(剩下的内容照常念)。
    #
    # 设 `taunt_reply: ""` 关掉这道闸。
    taunt_reply: str = "ザコザコ"
    taunt_words: list[str] = field(
        default_factory=lambda: ["杂鱼", "雑魚", "ザコ", "zako", "雜魚"])

    # 画面里哪个名牌 = **做出小mo 的人**(也就是主人)。
    #
    # 🔴 **必须给一组变体,而且在程序里做模糊匹配。** 视觉读名字**会读错** ——
    #    实测同一张名牌读出过 `莫西干 9aa3` / `夏西干9aa3` / `巽西干 9aa3 o`。
    #    指望大模型从描述里自己认出"这写的是莫西干"是不现实的,**它拿到的就是错字**。
    #    这里做归一化(去空格标点、统一小写)后的**包含匹配**,命中就明确告诉它。
    #    `9aa3` 是最稳的一段(错字主要错在汉字上,数字后缀很少错)。
    owner_names: list[str] = field(
        default_factory=lambda: ["莫西干9aa3", "莫西干", "9aa3"])

    # ---- 移动类动作 —— **只有主人能触发** ----
    #
    # 走的是和手势**同一套方括号标记**(LLM 吐 `[前进]`),但执行的是 /input/*
    # 真的让人物移动,而不是摆姿势。所以权限必须卡死:`_gate_permission()`
    # 拿这个列表判断,**非主人时直接把标记剥掉**。
    #
    # 「跟随」「过来」「停下」留给跟随服务(还没做),现在只记日志;
    # 「前进/后退/左移/右移」立刻生效(走 OSC `/input/*`)。
    move_actions: list[str] = field(default_factory=lambda: [
        "跟随", "过来", "停下",
        "前进", "后退", "左移", "右移",
    ])
    # 方向键每次按多久(秒)。`/input/*` 是**按住的语义,必须自己松开**,
    # 不松人物会一直走 —— 停不下来比不动更糟。0.45 ≈ 一步(见 tools/walk.py)。
    move_step_s: float = 0.45
    # 跟随服务的指令端口(`tools/follow_server.py` 在听)。
    # 「跟随」「过来」「停下」发到这里;方向键还是直接走 OSC。
    # 设 0 = 关掉(那几个动作就只记日志)。
    follow_port: int = 39601

    # 「谁在说话」的**视觉**判断(名牌发光描边,`SpeakerTracker`)。
    #
    # 🔴 **默认关掉** —— 两个原因,都不是"以后再修修看":
    #   ① **它从来没可靠过。** 发光描边的判据反复踩坑(底色随玩家变、
    #      旁边站个亮角色就假阳性、采样几何脆弱),实测没做成可用的东西。
    #   ② **它的活儿已经被别人干了**:转向说话的人靠的是
    #      **立体声判左右 + 名牌定角度**,跟这个没关系。
    #   它唯一的下游是"给记忆打说话人标签",而它基本恒返回 None,
    #   也就是**开不开行为一样,白烧 CPU**。
    #
    # ⚠️ 它每 3 秒跑一次 OCR,**是管线 CPU 的两大来源之一**(另一个是名牌定位)。
    #    实测两个 OCR 循环合计把管线推到 287% 一个核。
    #
    # 想再试一次视觉判说话人(比如换 OCR 文本框锚定)就设 True。
    speaker_track: bool = False

    # 「谁在说话」—— 靠名牌发光描边的视觉判断(src/vision/speaking.py)。
    # 结果会当成说话人标签写进记忆,检索时优先取本人的。
    #
    # ⚠️ **整条视觉链路是可选的**:找不到窗口就自动降级成"说话人未知",
    #    语音功能完全不受影响。所以默认开着是安全的。
    #    设 `vision_window: ""` 彻底关掉(不启动视觉线程,不装 OCR)。
    vision_window: str = "VRChat"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    osc: OscConfig = field(default_factory=OscConfig)
    chatbox: ChatboxConfig = field(default_factory=ChatboxConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    voiceprint: VoiceprintConfig = field(default_factory=VoiceprintConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    persona: str = "你是一个友好的 VRChat 虚拟伙伴。"
    config_path: Path = field(default=DEFAULT_CONFIG_PATH, repr=False)


# --------------------------------------------------------------------------
# 构造
# --------------------------------------------------------------------------

def _build(cls: type, data: Any):
    """从 dict 构造 dataclass,忽略多余字段(方便以后加配置项不报错)。"""
    if not isinstance(data, dict):
        return cls()
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config(
        llm=_build(LLMConfig, raw.get("llm")),
        osc=_build(OscConfig, raw.get("osc")),
        chatbox=_build(ChatboxConfig, raw.get("chatbox")),
        audio=_build(AudioConfig, raw.get("audio")),
        vad=_build(VadConfig, raw.get("vad")),
        asr=_build(AsrConfig, raw.get("asr")),
        tts=_build(TtsConfig, raw.get("tts")),
        wake=_build(WakeConfig, raw.get("wake")),
        memory=_build(MemoryConfig, raw.get("memory")),
        voiceprint=_build(VoiceprintConfig, raw.get("voiceprint")),
        behavior=_build(BehaviorConfig, raw.get("behavior")),
        # ⚠️ **新增配置段时别忘了在这里接一下** —— `scene` 加进 Config 时漏了这行,
        #    结果 config.yaml 里的 `scene:` 段**从来没被读过**,一直吃 dataclass 默认值。
        #    症状很隐蔽:程序不报错,只是"改了配置没反应"。
        scene=_build(SceneConfig, raw.get("scene")),
        config_path=path,
    )
    if raw.get("persona"):
        cfg.persona = str(raw["persona"]).strip()

    cfg.asr.device = str(cfg.asr.device).lower()
    return cfg


def validate(cfg: Config) -> list[str]:
    """返回一组人类可读的警告,不抛异常 —— 让调用方决定怎么处理。"""
    warn: list[str] = []

    if not os.environ.get(cfg.llm.api_key_env):
        warn.append(
            f"环境变量 {cfg.llm.api_key_env} 未设置,chat / run 会失败。"
            f"去 platform.deepseek.com 拿 key,然后 `setx {cfg.llm.api_key_env} sk-xxx`"
            f"(设完要重开终端)。"
        )

    if cfg.llm.model == "deepseek-reasoner":
        warn.append(
            "deepseek-reasoner 会先输出一大段推理再回答,语音对话里首句延迟会很高。"
            "闲聊建议用 deepseek-chat。"
        )

    cap = cfg.audio.capture_device
    if isinstance(cap, str) and "loopback" not in cap.lower() and "回环" not in cap:
        warn.append(
            f"capture_device 是 {cap!r},不是回环。桌面模式下 VRChat 音频走扬声器,"
            f"要听到玩家得填 \"loopback\"(或者 \"loopback:设备名片段\")。"
        )

    if cfg.audio.output_device is None:
        warn.append(
            "output_device 是 null,AI 会从你的扬声器说话 —— 你自己听得见,"
            "但 VRChat 里没人听得见。要真正接进去得装 VB-CABLE 并填 \"CABLE Input\"。"
        )

    if cfg.tts.backend == "cosyvoice":
        if not os.environ.get(cfg.tts.cloud_api_key_env):
            warn.append(
                f"tts.backend=cosyvoice 但环境变量 {cfg.tts.cloud_api_key_env} 没设,"
                f"合成会退回 edge(音色不是克隆的)。"
                f"`setx {cfg.tts.cloud_api_key_env} sk-xxx`(设完重开终端)。"
            )
        if not cfg.tts.cloud_voice_id:
            warn.append(
                "tts.backend=cosyvoice 但 tts.cloud_voice_id 是空的 —— "
                "先跑 tools/enroll_dashscope_voice.py 注册音色,把打印出来的 voice_id "
                "填进 config.yaml;在填之前合成会直接走 edge。"
            )

    return warn
