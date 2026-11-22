"""从视频/音频里做出一段可用于**音色复刻**的参考音频。

## 为什么要这个工具

云端声音复刻(百炼)要 **10~20s 干净音频**。但素材往往是游戏/番剧切片 ——
**一句一句的,凑不出连续 15 秒**。实测某段素材里最长连续语音只有 2.7s。

所以做法是**拼接多段短语音**。但拼接带来一个必须解决的问题:
**素材里可能不止一个人说话**(游戏对话、和声、音效)。拼错了 = 音色被稀释甚至跑偏。

## 怎么解决"是不是同一个人" —— 用项目里现成的 CAM++ 声纹

不靠耳朵逐段听(几十段听不过来),而是给每段算 CAM++ 声纹向量,再算
**「这段和其余所有段的平均相似度」**:

    高(>0.6)  → 主流说话人,可用
    中(0.2~0.4) → 另一个说话人
    接近 0     → **正交**,那不是人声(音效/BGM sting),不是"别人"

实测那条视频:37 段里 24 段 >0.60,两个 0.0 附近的确实是音效。
**"正交 ≠ 第二个人"这个区分很关键** —— 两者处理方式不同(音效要丢,第二个人要单独成簇)。

## 用法

    # 先下音轨(不需要系统 ffmpeg,yt-dlp 直接拿 B 站音频流)
    yt-dlp -f bestaudio -o source.m4a "https://www.bilibili.com/video/BVxxxx/"

    # 出参考音频
    .venv\\Scripts\\python.exe tools\\make_ref_audio.py source.m4a --out ref_voice_15s.wav

    # 调参
    --secs 15        目标时长(百炼要求 10~20)
    --min-sim 0.60   声纹过滤门限,调低会放进可疑段(脚本会警告)
    --list           只分析、不写文件(先看看素材够不够)

## 产物

`--out` 指定的 wav:**32kHz 单声道 PCM16、峰值归一 -3dBFS**。
32kHz 满足百炼的 ≥24kHz,和之前 GPT-SoVITS 那条路的参考音频保持一致。

⚠️ **工具只保证"同一个人 + 有声"**,保证不了"没有 BGM"。
   全程垫着伴奏的视频,人声段照样能过声纹闸。**出完必须人耳听一遍**。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ⚠️ **必须用 `src.audio.resample`(soxr),不能用 `src.audio.voiceprint.resample`**。
# 后者是手写的加窗 sinc,只在**降采样到 1/整数**(如 48k→16k,up=1)时正确;
# 一旦 up>1(如 48k→32k,up=2)它用 `mode="same"` 卷积后再按 down 抽取,
# 输出长度变成 len/3 而不是 len*2/3 —— **静默丢掉一半样本**。
# 项目里没人踩到,是因为调用方全是 48k→16k,正好绕过。
from src.audio.resample import resample  # noqa: E402
from src.audio.voiceprint import Voiceprint  # noqa: E402
from src.config import load  # noqa: E402

try:                                   # 中文控制台是 GBK,emoji 会直接崩
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_SR = 32000        # 百炼要 ≥24k;32k 与其它参考音频保持一致
VP_SR = 16000         # CAM++ 只吃 16k
FRAME_S = 0.1         # RMS 分帧粒度
MIN_SEG_S = 1.0       # 短于它的段不进声纹分析(embedding 太噪)
GAP_S = 0.12          # 拼接时段间的静音,避免韵律突变听感发假
FADE_S = 0.01         # 每段两端淡入淡出,防拼接点爆音
PEAK = 0.707          # 归一到 -3dBFS


def decode(path: Path) -> tuple[np.ndarray, int]:
    """解出单声道 float32。用 PyAV —— 它自带 ffmpeg 库,不需要系统装 ffmpeg。

    (本项目已经因为 faster-whisper 装了 av,所以这是零新增依赖。)
    """
    import av
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        chunks = [f.to_ndarray() for f in c.decode(st)]
        sr = st.sample_rate
    if not chunks:
        raise SystemExit(f"❌ 解不出音频:{path}")
    x = np.concatenate(chunks, axis=1) if chunks[0].ndim == 2 else np.concatenate(chunks)
    if x.ndim == 2:
        x = x.mean(axis=0)                 # 立体声 → 单声道
    x = x.astype(np.float32)
    if np.abs(x).max() > 1.5:
        x = x / 32768.0                    # 整数 PCM → [-1,1]
    if sr != OUT_SR:
        x = resample(x, sr, OUT_SR).astype(np.float32)
        sr = OUT_SR
    return x, sr


def voiced_runs(x: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """找出连续有声段,返回 [(起点秒, 终点秒)]。

    门限用**自适应**的:取 RMS 分布的低分位当噪声底,底噪 +15dB 当门限。
    固定门限在不同素材上(录音电平差很多)会失灵 —— 这条视频峰值才 0.264。
    """
    hop = int(FRAME_S * sr)
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    thr = max(np.percentile(db, 30) + 15.0, -55.0)
    v = db > thr
    runs, i = [], 0
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            if (j - i) * FRAME_S >= MIN_SEG_S:
                runs.append((i * FRAME_S, j * FRAME_S))
            i = j
        else:
            i += 1
    return runs


def score_segments(x: np.ndarray, sr: int, runs: list[tuple[float, float]]) -> list[dict]:
    """给每段算 CAM++ 声纹,再算「和其余各段的平均相似度」。

    这个"平均相似度"是**相对量**,不是绝对门限 —— 它回答的是
    「这段像不像这条素材的主流说话人」,正好是拼接参考音频要问的问题。
    """
    vp = Voiceprint(load().voiceprint)
    segs = []
    for a, b in runs:
        seg = x[int(a * sr): int(b * sr)]
        e = vp.embed(resample(seg.astype(np.float32), sr, VP_SR))
        segs.append({"a": a, "b": b, "len": b - a, "emb": e})
    if len(segs) < 2:
        for s in segs:
            s["sim"] = 1.0
        return segs, vp
    E = np.stack([s["emb"] for s in segs])
    S = E @ E.T
    np.fill_diagonal(S, np.nan)
    for i, s in enumerate(segs):
        s["sim"] = float(np.nanmean(S[i]))
    return segs, vp


def speaker_ref(segs: list[dict]) -> np.ndarray:
    """把"主流说话人"压成一个参考向量 —— 给 best_window 的声纹闸用。

    取相似度最高的那批段求平均。不取全部:低的那些本来就可疑(音效/另一个人),
    混进来会把参考向量拉歪。
    """
    top = sorted(segs, key=lambda s: -s["sim"])[: max(3, len(segs) // 4)]
    v = np.mean([s["emb"] for s in top], axis=0)
    return v / (np.linalg.norm(v) or 1.0)


def build(x: np.ndarray, sr: int, segs: list[dict], secs: float) -> tuple[np.ndarray, list[dict]]:
    """按声纹可信度从高到低取段,拼到目标时长。

    优先取长段(同样可信度下,长段信息更足),所以排序键是 (sim 分档, 长度)。
    """
    pool = sorted(segs, key=lambda s: (round(s["sim"], 1), s["len"]), reverse=True)
    cut, used, total = int(FADE_S * sr), [], 0.0
    for s in pool:
        if total >= secs:
            break
        seg = x[int(s["a"] * sr): int(s["b"] * sr)].astype(np.float32)
        seg = seg.copy()
        f = min(cut, len(seg) // 2)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            seg[:f] *= ramp
            seg[-f:] *= ramp[::-1]
        used.append(s)
        total += s["len"]
    # 用静音串起来
    gap = np.zeros(int(GAP_S * sr), dtype=np.float32)
    out = []
    for i, s in enumerate(used):
        if i:
            out.append(gap)
        out.append(x[int(s["a"] * sr): int(s["b"] * sr)].astype(np.float32))
    y = np.concatenate(out) if out else np.zeros(1, np.float32)
    if len(y) > int((secs + 1.0) * sr):        # 别超太多
        y = y[: int(secs * sr)]
    peak = float(np.abs(y).max()) or 1.0
    return (y / peak * PEAK).astype(np.float32), used


def best_window(x: np.ndarray, sr: int, secs: float, vp=None, ref_emb=None,
                min_sim: float = 0.60, max_gap_s: float = 1.2):
    """找一段**连续**的高密度语音窗口 —— 给 GPT-SoVITS 用。

    和拼接模式(`build`)的区别,两条路的约束是**相反**的:

      · 百炼(云端复刻):要 10~20s。素材是一句一句的话、凑不出连续 15s,**只能拼接**。
      · GPT-SoVITS(zero-shot):要一段**自然连续**的 3~10s。拼接在这里是坏的 ——
        接缝处韵律会断,而且它还要一句 `prompt_text`(参考音频里**真实说的那句话**),
        拼了四五段就没法给出对应的文字了。

    ## 🔴 为什么这里也必须要声纹闸

    **踩过的坑**:这函数一开始只按 RMS 能量找窗口。结果它选了整条素材里密度最高
    的一段 —— 但那是一段**音效**,声纹相似度只有 0.026(正交)。
    纯能量检测**分不出"人在说话"和"有声响"**,所以必须再拿声纹确认它像主流说话人。

    (那个 0.026 是"正交"而不是"另一个人" —— 换人通常在 0.2~0.4。)

    返回 (起点秒, 时长秒, 有声占比);找不到返回 None。
    """
    hop = int(FRAME_S * sr)
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    v = db > max(np.percentile(db, 30) + 15.0, -55.0)

    # 时长在 [3, 10] 里从长到短试 —— 长一点音色更足,但静音更容易超标
    for s_len in (secs, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0):
        if not (3.0 <= s_len <= 10.0):
            continue
        W = int(s_len * 10)
        cands = []
        for s in range(0, n - W + 1):
            w = v[s:s + W]
            mx = cur = 0
            for b in w:                      # 窗口内最长连续静音
                cur = 0 if b else cur + 1
                mx = max(mx, cur)
            if mx * FRAME_S <= max_gap_s:
                cands.append((w.mean(), s * FRAME_S, s_len))
        if not cands:
            continue
        cands.sort(reverse=True)
        for ratio, st, ln in cands[:8]:      # 密度前 8 名里挑,逐个过声纹
            if vp is None or ref_emb is None:
                return st, ln, ratio
            seg = x[int(st * sr): int((st + ln) * sr)].astype(np.float32)
            sim = float(vp.embed(resample(seg, sr, VP_SR)) @ ref_emb)
            if sim >= min_sim:
                return st, ln, ratio
            print(f"  ⚠️ {st:.1f}s({ln:.1f}s) 密度 {ratio*100:.0f}% 但声纹只有 "
                  f"{sim:.3f} —— 不是主流说话人(多半是音效),跳过")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="从视频/音频做音色复刻的参考音频")
    ap.add_argument("src", type=Path, help="源音频/视频文件(m4a/mp4/wav/...)")
    ap.add_argument("--out", type=Path, help="输出的 wav 路径")
    ap.add_argument("--secs", type=float, default=15.0, help="目标时长(百炼要 10~20)")
    ap.add_argument("--min-sim", type=float, default=0.60, help="声纹过滤门限")
    ap.add_argument("--list", action="store_true", help="只分析不写文件")
    ap.add_argument("--single", action="store_true",
                    help="取**一段连续**音频而不是拼接(GPT-SoVITS 要这种;百炼要拼接的那种)")
    a = ap.parse_args()

    if not a.src.exists():
        raise SystemExit(f"❌ 找不到 {a.src}")
    if not (10.0 <= a.secs <= 20.0) and not a.list:
        print(f"⚠️ --secs {a.secs} 不在百炼要求的 10~20s 区间,大概率被拒")

    print(f"解码 {a.src.name} ...")
    x, sr = decode(a.src)
    print(f"  {len(x)/sr:.1f}s @ {sr}Hz, 峰值 {np.abs(x).max():.3f}")

    runs = voiced_runs(x, sr)
    print(f"可用语音段(≥{MIN_SEG_S}s): {len(runs)} 段, 合计 {sum(b-a for a,b in runs):.1f}s")
    if not runs:
        raise SystemExit("❌ 没找到语音段 —— 素材可能全程静音或门限不对")

    print("算声纹(CAM++) ...")
    segs, vp = score_segments(x, sr, runs)
    segs.sort(key=lambda s: -s["sim"])
    print(f"\n{'起点':>8} {'时长':>6} {'声纹相似':>8}  判定")
    for s in segs[:20]:
        mark = "✅ 用" if s["sim"] >= a.min_sim else ("⚠️ 疑似别人" if s["sim"] > 0.15 else "❌ 非人声(音效?)")
        print(f"{s['a']:7.1f}s {s['len']:5.1f}s {s['sim']:8.3f}  {mark}")
    if len(segs) > 20:
        print(f"  ... 另有 {len(segs)-20} 段未列")

    if a.single:
        win = best_window(x, sr, a.secs, vp=vp, ref_emb=speaker_ref(segs),
                          min_sim=a.min_sim)
        if win is None:
            raise SystemExit("❌ 找不到同时满足「静音<1.2s」和「声纹像主流说话人」的连续窗口")
        st, ln, ratio = win
        print(f"\n--single: 最优连续窗口 {st:.1f}s ~ {st+ln:.1f}s "
              f"({ln:.1f}s, 有声 {ratio*100:.0f}%)")
        if not a.out:
            raise SystemExit("❌ 没给 --out")
        seg = x[int(st * sr): int((st + ln) * sr)].astype(np.float32).copy()
        f = min(int(FADE_S * sr), len(seg) // 2)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            seg[:f] *= ramp
            seg[-f:] *= ramp[::-1]
        y = (seg / (np.abs(seg).max() or 1.0) * PEAK).astype(np.float32)
        import soundfile as sf
        a.out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(a.out), y, sr, subtype="PCM_16")
        print(f"\n✅ 写出 {a.out}")
        print(f"   {len(y)/sr:.1f}s @ {sr}Hz PCM16, 峰值 {np.abs(y).max():.3f}")
        print("\n⚠️ **GPT-SoVITS 还要一句 `prompt_text`** —— 这段里真实说的那句话,")
        print("   写错音色会跑偏。用项目自己的 ASR 转一遍就有了。")
        print("⚠️ 并且必须人耳听一遍:是不是同一个人、有没有 BGM。")
        return 0

    good = [s for s in segs if s["sim"] >= a.min_sim]
    have = sum(s["len"] for s in good)
    print(f"\n过闸({a.min_sim}): {len(good)}/{len(segs)} 段, 合计 {have:.1f}s (需要 {a.secs:.0f}s)")
    if have < a.secs:
        raise SystemExit(
            f"❌ 过闸素材不够拼出 {a.secs:.0f}s。"
            f"要么换更长的素材,要么降 --min-sim(但会放进可疑段,必须人耳复核)"
        )
    if a.list:
        return 0
    if not a.out:
        raise SystemExit("❌ 没给 --out")

    y, used = build(x, sr, good, a.secs)
    import soundfile as sf
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(a.out), y, sr, subtype="PCM_16")

    print(f"\n✅ 写出 {a.out}")
    print(f"   {len(y)/sr:.1f}s @ {sr}Hz PCM16, 峰值 {np.abs(y).max():.3f} (-3dBFS)")
    print(f"   用了 {len(used)} 段:" + " ".join(f"{s['a']:.1f}s" for s in used))
    print("\n⚠️ 现在**必须戴耳机听一遍**:①是不是同一个人 ②有没有 BGM/伴奏 ③有没有爆音")
    print("   工具只保证了「同一个人 + 有声」,保证不了「没有伴奏」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
