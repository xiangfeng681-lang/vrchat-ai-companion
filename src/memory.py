"""向量记忆:让小mo 记得住以前聊过的事。

**为什么需要**:LLM 的上下文只有 12 轮(`history_turns`),超出就丢掉,
进程一重启更是全忘 —— 它甚至不记得你是谁。向量记忆把历史对话存进库,
每轮按**语义相关度**捞几条塞回上下文,于是"你上次说你喜欢崩坏系列"这种
跨越几十轮、甚至跨越几天的信息还能被想起来。

**为什么不用专业向量库**:几千条记忆下,一次全量余弦相似度只要几毫秒。
SQLite(标准库自带)+ numpy 就够了,少一个依赖少一份故障。

**为什么嵌入走 CPU**:显存已经被 VRChat + Whisper + GPT-SoVITS 吃满
(8GB 卡的余量只剩几百 MB)。BGE-small-zh 的 int8 量化版只有 24MB,
跑 CPU 上给一句话算向量约 5~15ms,完全够用 —— 记忆检索不在关键路径上,
它发生在"转写完、调 LLM 之前",那段时间本来就在等网络。

模型文件在 `models/bge-small-zh/`,从 `Xenova/bge-small-zh-v1.5` 下的
ONNX 量化版(int8)。**只依赖 onnxruntime + tokenizers,不需要 torch。**
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "bge-small-zh"

# BGE 中文模型检索时的官方建议:给**查询**加这段指令,文档不加。
# v1.5 里它是可选的,但加上检索效果更稳。
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class VectorMemory:
    """对话记忆的存取。embedding 走 ONNX(CPU),存储走 SQLite。"""

    def __init__(self, cfg, *, db_path: str | Path | None = None):
        from .config import MemoryConfig  # 避免循环导入

        self.cfg: MemoryConfig = cfg
        self.db_path = Path(db_path) if db_path else ROOT / cfg.db_path
        self._sess = None
        self._tok = None
        self._conn: sqlite3.Connection | None = None

    # ---------------- 模型 ----------------

    def _ensure_model(self):
        """惰性加载 —— 没开记忆功能就一个字节都不读。"""
        if self._sess is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok_path = MODEL_DIR / "tokenizer.json"
        onnx_path = MODEL_DIR / "model_quantized.onnx"
        if not tok_path.exists() or not onnx_path.exists():
            raise RuntimeError(
                f"嵌入模型不在 {MODEL_DIR} —— 需要 model_quantized.onnx 和 tokenizer.json。"
                "从 hf-mirror.com/Xenova/bge-small-zh-v1.5 下 onnx/model_quantized.onnx、"
                "tokenizer.json、config.json。"
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.enable_truncation(max_length=self.cfg.max_tokens)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        # 嵌入不是瓶颈,别让它跟 Whisper 抢满 CPU
        opts.intra_op_num_threads = max(1, self.cfg.cpu_threads)
        opts.log_severity_level = 3          # 只报错,别刷 warning
        self._sess = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        """一批文本 → L2 归一化的向量矩阵 (n, dim)。

        BGE 是 BERT 系,取 last_hidden_state 后做**带掩码的均值池化** ——
        不能直接 mean,否则 padding 的 0 会把向量拉偏。
        """
        self._ensure_model()
        encs = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        names = {i.name for i in self._sess.get_inputs()}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)

        hidden = self._sess.run(None, feeds)[0]            # (n, T, dim)
        m = mask[..., None].astype(hidden.dtype)           # (n, T, 1)
        summed = (hidden * m).sum(axis=1)                  # (n, dim)
        counts = np.clip(m.sum(axis=1), 1e-6, None)        # (n, 1)
        vecs = summed / counts
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    # ---------------- 存储 ----------------

    def _ensure_db(self):
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts REAL NOT NULL,
                   user_text TEXT NOT NULL,
                   assistant_text TEXT NOT NULL DEFAULT '',
                   embedding BLOB NOT NULL,
                   speaker TEXT
               )"""
        )
        # **迁移老库** —— `speaker` 是 2026-09-17 才加的,之前建的库没有这一列。
        # 直接 ALTER 加,NULL = "不知道谁说的"(老数据都是单人时期录的)。
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)")}
        if "speaker" not in cols:
            self._conn.execute("ALTER TABLE memories ADD COLUMN speaker TEXT")
        self._conn.commit()

    def add(self, user_text: str, assistant_text: str = "",
            speaker: str | None = None) -> None:
        """记一轮对话。太短的直接跳过 —— 「嗯」「好的」存进去只会污染检索。

        `speaker` 是归一化后的说话人名(见 `vision.speaking.norm_name`)。
        **认不出是谁就传 None** —— 宁可标记成"不知道",也别猜一个安上去。
        """
        user_text = (user_text or "").strip()
        if len(user_text) < self.cfg.min_chars:
            return
        self._ensure_db()
        vec = self._embed([user_text])[0]
        self._conn.execute(
            "INSERT INTO memories (ts, user_text, assistant_text, embedding, speaker)"
            " VALUES (?,?,?,?,?)",
            (time.time(), user_text, (assistant_text or "").strip(),
             vec.astype(np.float32).tobytes(), speaker),
        )
        self._conn.commit()
        self._prune()

    def _prune(self) -> None:
        """超过上限就删最旧的。防止库无限长大。"""
        n = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if n > self.cfg.max_entries:
            self._conn.execute(
                "DELETE FROM memories WHERE id IN "
                "(SELECT id FROM memories ORDER BY ts ASC LIMIT ?)",
                (n - self.cfg.max_entries,),
            )
            self._conn.commit()

    def recall(self, query: str, k: int | None = None,
               speaker: str | None = None) -> list[str]:
        """按语义相关度捞最近的若干条记忆,返回可读的字符串列表。

        低于相似度门限的丢掉 —— 宁可不给,也别塞一堆不相关的东西进去
        污染上下文(那比没有记忆更糟)。

        ## 多人场景:`speaker` 是"这句话是谁说的"

        **两轮取**:
          1. 先只捞**说话人本人**的(他上次说的话,对他才有意义)
          2. 本人这边捞不够 `k` 条,才**放宽到所有人**

        为什么不是硬过滤:硬过滤会让"新来的人"面前小mo 一片空白 —— 连常识性的
        旧事都想不起来,表现是"突然变傻"。**放宽一轮,退化成今天的行为,不会更差。**

        ⚠️ `speaker=None` 表示**认不出是谁**(不在画面里/没发光)。这种情况
        不做任何区分,直接全员检索 —— 不知道是谁就别乱猜。
        """
        query = (query or "").strip()
        if len(query) < self.cfg.min_chars:
            return []
        self._ensure_db()
        rows = self._conn.execute(
            "SELECT ts, user_text, assistant_text, embedding, speaker FROM memories"
        ).fetchall()
        if not rows:
            return []

        qv = self._embed([_QUERY_PREFIX + query])[0]
        mat = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        sims = mat @ qv                                     # 都是单位向量,点积即余弦

        n = k or self.cfg.top_k
        order = np.argsort(-sims)

        def take(idx_iter, limit: int, thresh: float) -> list[str]:
            got: list[str] = []
            for i in idx_iter:
                if sims[i] < thresh:
                    break
                ts, u, a, _, sp = rows[int(i)]
                when = time.strftime("%m-%d %H:%M", time.localtime(ts))
                # 认得出来就写名字,认不出来还是写"用户" —— 别编
                who = sp or "用户"
                line = f"[{when}] {who}说:{u}"
                if a:
                    line += f" | 你当时回答:{a[:60]}"
                got.append(line)
                if len(got) >= limit:
                    break
            return got

        if not speaker:
            return take(order, n, self.cfg.min_similarity)

        # 🔑 **本人这一轮的门槛要更低。** 踩过的坑:实测「我讨厌吃鱼」对查询
        # 「我喜欢吃什么」只有 0.435,低于全局门限 0.45 —— 于是**本人的记忆
        # 在进入说话人逻辑之前就被筛掉了**,只能放宽去找别人的,
        # 正好变成我们想避免的"把 B 当成小明"。
        # 道理:只筛本人时候选本来就少,他**自己**的 0.435 也比陌生人的 0.492 更相关。
        own_thresh = self.cfg.min_similarity * 0.8
        own = take((i for i in order if rows[int(i)][4] == speaker), n, own_thresh)
        if len(own) >= n:
            return own
        # 本人这边不够 → 放宽。已经捞到的排前面(本人优先),后面的补其他人。
        rest = take((i for i in order if rows[int(i)][4] != speaker), n - len(own),
                    self.cfg.min_similarity)
        return own + rest

    def warmup(self) -> None:
        """提前把嵌入模型加载起来。启动时调一次,免得第一轮对话多等一秒。"""
        self._ensure_model()

    def count(self) -> int:
        self._ensure_db()
        return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
