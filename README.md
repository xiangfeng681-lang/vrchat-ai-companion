# vrchat-ai-companion

Real-time AI companion for VRChat: voice chat, vision, voiceprint, memory, following and dancing —
all over the official OSC interface.

VRChat 实时 AI 伙伴：语音对话、视觉、声纹、记忆、跟随、跳舞，全部走官方 OSC 接口。

> 本仓库不含任何音色模型或参考音频（见「音色是怎么做的」）。

## 功能

| 模块 | 做什么 |
|---|---|
| 耳朵 | WASAPI 回环抓 VRChat 输出 → VAD 切句 → ASR（云端千问 / 本地 faster-whisper） |
| 大脑 | 千问（百炼 OpenAI 兼容接口），流式按句吐出，带人设与话轮控制 |
| 嗓子 | TTS 三选一：`gpt_sovits` 本地克隆 / `cosyvoice` 云端克隆 / `edge` |
| 嘴和脸 | OSC `/chatbox/input` 字幕 + `/avatar/parameters` 口型 |
| 眼睛 | 窗口抓取（WGC）→ 名牌 YOLO + OCR / 场景描述 |
| 认人 | CAM++ 声纹，判断"是不是主人说的"，权限分级 |
| 记忆 | 向量记忆（BGE-small-zh）+ SQLite |
| 跟随 | 视觉锁定玩家，语音/OSC 驱动移动 |
| 动作 | AnyaDance 协议：idle 姿势、手势、整段舞蹈 |

## 架构

```
VRChat 扬声器 ──WASAPI 回环──► VAD ──► ASR ──► LLM（按句流式）
                                                  │
        ┌─────────────────────────────────────────┘
        ▼
      TTS（本地克隆 / 云端克隆 / edge）──► VB-Audio Cable ──► VRChat 麦克风
        │
        └─► OSC：聊天框字幕 + 口型参数
截图（WGC）──► YOLO/OCR/场景描述 ──► 说话人识别 / 跟随 / 认主人
```

## 技术路线

| 层 | 技术选型 |
|---|---|
| **采集** | WASAPI 回环（`pyaudiowpatch`，抓的是数字音频流而不是麦克风）、`sounddevice`、`soxr` 重采样 |
| **切句** | 自实现能量 VAD（RMS 双阈值 + 停顿判定），没用 WebRTC / Silero |
| **语音识别** | 云端千问全模态（百炼）/ 本地 `faster-whisper`（CTranslate2）兜底 |
| **视觉** | `windows-capture`（Windows.Graphics.Capture）→ YOLO（ONNX Runtime）找名牌 → `RapidOCR` 读字 → 千问 VL 描述场景 |
| **声纹** | CAM++（`sherpa-onnx`，纯 CPU）判断"是不是主人" |
| **认知** | 通义千问（OpenAI 兼容接口），流式按句返回；人设 + 话轮控制 |
| **记忆** | BGE-small-zh 句向量 + SQLite 向量检索 |
| **合成** | GPT-SoVITS（本地 zero-shot 克隆）/ CosyVoice（云端克隆）/ edge-tts |
| **输出** | VRChat OSC：`/chatbox/input` 字幕 + `/avatar/parameters` 口型 |
| **动作** | AnyaDance 协议：idle 姿势、手势、整段舞蹈 |
| **并发** | 多线程 + `queue`（不是 asyncio：采集/视觉/合成各自阻塞，线程模型更直观） |
| **配置** | dataclass + YAML，缺项走默认值，启动时统一校验 |

## 工程难点与解法

这些都是实际踩过、并且在代码里留下实现的：

| 问题 | 根因 | 解法 |
|---|---|---|
| **抓不到 VRChat 画面** | `PrintWindow` 对 DXGI flip-model 窗口失效，四种 flag 全返回 0 | 改用 **Windows.Graphics.Capture**；常驻流 55 Hz（一次性抓取要 0.31 s） |
| **声纹把主人认成陌生人** | 回环是混音，按中位数算只有 0.228 | 改 **top-2 均值**，实测 0.692，阈值 0.42 |
| **记忆答非所问** | 该嵌入模型分不开相关与不相关（相关 0.435 / 不相关 0.569），调 `min_similarity` 没用 | 改**提示词硬规则**：「先回答本轮问题，往事不相关就整轮别提」 |
| **机器人自己叫醒自己** | 视觉读到自己头顶的 OSC 字幕 | 加**句读闸**：带标点的文本块判为气泡，不算名牌 |
| **千问首句 71 秒** | 推理模型把 token 全花在思考上 | `enable_thinking: false` → **3 秒** |
| **云端 ASR 没有热词** | 兼容接口不支持 hotwords | `cloud_hint` 把专有名词拼进上下文补偏置 |
| **AnyaDance 整帧被拒** | 帧里缺 `rotation_xyzw` | 补全四元数；手编坐标全部重算（手位差 45cm / 臂长 0.505 硬约束） |
| **MMD 动作对不上** | 绝对坐标与 VRChat 骨骼不兼容 | 改用**相对基准帧位移** + 左右翻转修正 |
| **VRChat 吃满 GPU** | 其实瓶颈在地图 | 换轻量世界 90% → 22%；顺手修掉抓图回调每帧拷贝（CPU 512% → 287%） |

## 设计取舍

| 决策 | 为什么这么选 | 代价 |
|---|---|---|
| **多线程 + queue**，不用 asyncio | 采集、抓帧、合成全是阻塞式 C 扩展，塞进 asyncio 还得包 executor；线程模型更直观，出问题好定位 | 共享状态要自己加锁 |
| **ASR 迁到云端**（保留本地兜底） | 省约 2 GB 显存，不再和 VRChat 抢 GPU；云端 0.4 s、自带标点 | 断网就哑，靠本地 `faster-whisper` 兜底 |
| **TTS 选云端 CosyVoice** | 本地 GPT-SoVITS 要 2~3 GB 显存（RTF≈0.48）；edge 有 ~1.85 s 固定开销且不能克隆音色 | 云端断开就没声，且在云端注册 10~20 s 参考音频 |
| **视觉只在唤醒时看** | 视觉链路 3~4 s，比首句慢；常驻看约 13 万 token/小时 | 唤醒那一轮常常赶不上第一句 |
| **声纹 + 视觉两条身份通道** | 声纹判"是不是主人"（决定权限），视觉判"谁在说话、在哪"（跟随）；都不完美，互相补 | 两套都要维护，且都可能误判 |
| **记忆靠提示词硬规则，不调相似度阈值** | 实测该嵌入模型分不开相关/不相关，提高阈值会误杀真相关的往事 | 规则是人写的，覆盖面有限 |
| **人设"只得意、不贬低"** | 管线拿不到说话人身份，贬低会误伤旁人 | 人设浓度受限，需要别的办法补 |
| **自己写能量 VAD** | 回环音频干净，能量判断够用；少一个模型依赖 | 嘈杂环境下不如 Silero |

## 稳定性与延迟

- 端到端压测：**5 小时 / 707 个事件 / 0 错误 / 0 崩溃**（2026-10-14）
- 首句中位 **3.14 s**；YOLO 34 ms/帧、OCR ≈1 s、VL ≈1.4 s
- 全部云端化后显存 **7600 MB → 2623 MB**，把显卡还给 VRChat

## 目录

```
main.py                入口：devices / check / ear / chat / asr / tts / osc / run
src/
  audio/              采集、设备、重采样、VAD、声纹
  vision/             窗口抓取、名牌检测、场景、说话指示、目标锁定
  asr.py llm.py tts.py memory.py osc_client.py pipeline.py config.py
tools/                跟随服务、跳舞、MMD 解算、音色注册、标注、诊断
config.example.yaml   配置模板（复制成 config.yaml 再改）
```

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
copy config.example.yaml config.yaml
setx DASHSCOPE_API_KEY sk-xxx      :: 大脑/耳朵/眼睛/云端音色共用一把 key
python main.py devices             # 看音频设备，填进 config.yaml
python main.py check               # 检查配置
python main.py run                 # 跑全流程
```

VRChat 里要打开 **OSC**（Action Menu → Options → OSC → Enabled）。

## 音色是怎么做的

**核心：只给一段参考音频，模型 zero-shot 复制音色，不训练。**

1. **准备参考音频** —— 5~10 秒、干净、单人、无 BGM。来源必须是你有权使用的素材。
2. **本地克隆（GPT-SoVITS）**，`config.yaml` 里：
   - `tts.backend: gpt_sovits`
   - `ref_audio`: 参考音频路径
   - `prompt_text`: 参考音频里**真实说的那句话**，必须一字不差，写错音色会跑偏
   - `speed_factor`: >1 变快（注意是重采样，会等比升调）
   - 需要先起 GPT-SoVITS 的 `api_v2.py` 服务（独立 venv，不在本仓库 requirements 里）
3. **云端克隆（百炼 CosyVoice）**：
   - `tools/make_ref_audio.py`：把零散短语音拼成 10~20 秒（用声纹判断是不是同一个人）
   - `tools/enroll_dashscope_voice.py`：注册得到 `voice_id`，填进 `config.yaml`
   - 合成模型必须和注册时**逐字一致**，音色不能跨模型用
4. **换音色**：换 `ref_audio` + `prompt_text` 即可，不用重训。

> 本仓库不含任何音色模型、参考音频或语音素材。示例配置里的音色参数已清空/占位。

## 模型文件（不含在仓库里）

按需自行准备，放在 `models/`：

| 路径 | 用途 |
|---|---|
| `models/bge-small-zh/` | 向量记忆嵌入（Xenova/bge-small-zh-v1.5） |
| `models/speaker/campplus_zh.onnx` | 声纹识别（CAM++） |
| `models/speaker/owner.npy` | 主人声纹注册结果 |
| `models/nameplate/nameplate.onnx` | 名牌检测（用 `tools/autolabel.py` 标注后训练） |

## 已知限制

- **仅 Windows**：WASAPI 回环、WGC 窗口抓取、OCR 都依赖 Windows
- 云端 ASR/LLM/音色都要 `DASHSCOPE_API_KEY`，断网就哑
- 视觉链路有延迟（截图 + OCR/OCR 比首句慢），唤醒那一轮常常赶不上
- VRChat 必须开 OSC，否则收不到也发不出

## 许可

MIT，见 [LICENSE](LICENSE)。
