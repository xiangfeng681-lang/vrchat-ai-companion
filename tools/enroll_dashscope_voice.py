"""把一段参考音频注册成阿里云百炼的云端音色,拿到 `voice_id`。

## 为什么需要"注册"这一步

本地 GPT-SoVITS 是 **zero-shot**:每次合成现场喂一遍参考音频。
云端百炼**不是** —— 它先把参考音频注册成一个音色 ID,之后合成只传这个 ID。
所以换音色 = 重新注册一次,拿个新 ID 贴进 `config.yaml`。

注册是**免费且可重复**的(不是训练),不满意就重跑。

## 参考音频的要求(实测,不是抄文档)

- **10~20 秒**。低于 10s 会被拒
- **≥24kHz**。`tools/make_ref_audio.py` 出的是 32k,合规
- **干净**:单人、无 BGM、无明显噪声。素材是游戏切片的话**必须拼接**
  (一句一句的话凑不出连续 15 秒),见 `tools/make_ref_audio.py`
- 单声道 WAV,base64 后 < 10MB(15s@32k ≈ 1.3MB,很宽裕)

## 🔴 两个路径不能混用(这是踩过的坑)

    官方 SDK 路径: model=voice-enrollment, action=create_voice, 参数 prefix
    Qwen 路径    : model=qwen-voice-enrollment, action=create, 参数 preferred_name

混用会报 `preprocess service not found` 之类。而且 **Qwen 路径走不通** ——
实测 `target_model=qwen3-tts-vc-2026-01-22` 能注册成功,但**没有任何合成端点
提供那个模型**(HTTP 端点、官方 SDK 的 MultiModalConversation 全是
`Model not exist`)。所以这里**只走 SDK 路径(CosyVoice 系列)**。

另一个实测结论:**`url` 字段可以塞 base64 data URI**,不需要公网可访问的地址。
官方文档和 SDK 签名都写 `url`,看起来像必须要公网 URL,实测不用 ——
这对家里 NAT 后面的机器很关键。

## 用法

    setx DASHSCOPE_API_KEY sk-xxx        # 设完重开终端
    .venv\\Scripts\\python.exe tools\\make_ref_audio.py <视频> --out ref_yy_15s.wav
    .venv\\Scripts\\python.exe tools\\enroll_dashscope_voice.py ref_yy_15s.wav

    # 换前缀(音色名前缀,只能小写字母和数字,≤10 字符)
    --prefix myvoice
    # 换绑定的合成模型(⚠️ 必须和 config.yaml 的 tts.cloud_model 一致)
    --target-model cosyvoice-v2

## 换音色 / 清理

    # 列出已注册的音色
    python tools/enroll_dashscope_voice.py --list
    # 删掉某个
    python tools/enroll_dashscope_voice.py --delete cosyvoice-v2-xxx-<hex>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ENROLL_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
MIN_SECS, MAX_SECS = 10.0, 20.0
MIN_SR = 24000
MAX_BYTES = 10 * 1024 * 1024


def _call(payload: dict, key: str, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        ENROLL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"❌ 百炼报错 {exc.code}: {detail}")
        if "preprocess service not found" in detail:
            print("   → 这条基本就是 **model/action 参数混用了两个路径**(见文件头说明)")
        if "Audio" in detail or "Decoder" in detail:
            print("   → 这条是**音频本身不合格**:太短/太吵/格式不对,先跑 make_ref_audio.py")
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        raise SystemExit(f"❌ 连不上百炼:{exc}(检查网络/代理)")


def _check_wav(p: Path) -> tuple[float, int]:
    """本地先把明显不合格的挡掉 —— 省一次往返,报错也更清楚。"""
    if not p.exists():
        raise SystemExit(f"❌ 找不到 {p}")
    import soundfile as sf
    info = sf.info(str(p))
    dur, sr = info.duration, info.samplerate
    print(f"参考音频: {p.name}  {dur:.1f}s  {sr}Hz  {info.channels}声道  "
          f"{info.subtype}  {p.stat().st_size/1024:.0f}KB")
    if not (MIN_SECS <= dur <= MAX_SECS):
        print(f"⚠️ 时长 {dur:.1f}s 不在百炼要求的 {MIN_SECS:.0f}~{MAX_SECS:.0f}s 区间,大概率被拒")
    if sr < MIN_SR:
        print(f"⚠️ 采样率 {sr}Hz 低于要求的 {MIN_SR}Hz,大概率被拒")
    if info.channels > 1:
        print("⚠️ 不是单声道 —— 百炼偏好单声道,建议先转")
    if p.stat().st_size > MAX_BYTES:
        raise SystemExit(f"❌ 文件 {p.stat().st_size/1e6:.1f}MB 超过 10MB 上限")
    return dur, sr


def enroll(wav: Path, target_model: str, prefix: str, key: str) -> dict:
    raw = wav.read_bytes()
    uri = "data:audio/wav;base64," + base64.b64encode(raw).decode()
    print(f"base64 后 {len(uri)/1024:.0f}KB,上传中…")
    body = _call({
        "model": "voice-enrollment",        # ⚠️ 不是 qwen-voice-enrollment
        "input": {
            "action": "create_voice",       # ⚠️ 不是 create
            "target_model": target_model,   # 🔴 必须和 config.yaml 的 tts.cloud_model 一致
            "prefix": prefix,               # ⚠️ Qwen 路径那个参数叫 preferred_name
            "url": uri,                     # 实测:这里塞 base64 data URI 也可以,不用公网 URL
        },
    }, key)
    vid = (body.get("output") or {}).get("voice_id") or ""
    if not vid:
        raise SystemExit(f"❌ 没拿到 voice_id,返回体: {json.dumps(body, ensure_ascii=False)[:500]}")
    return {
        "voice_id": vid,
        "target_model": target_model,
        "prefix": prefix,
        "ref_audio": wav.name,
        "ref_secs": round(_check_wav(wav)[0], 1),
        "ref_sha256": hashlib.sha256(raw).hexdigest()[:16],
    }


def main() -> int:
    cfg = load()
    ap = argparse.ArgumentParser(description="注册百炼云端音色")
    ap.add_argument("wav", nargs="?", type=Path, help="参考音频(10~20s, ≥24kHz)")
    ap.add_argument("--prefix", default="xiaomo", help="音色名前缀,小写字母数字,≤10字符")
    ap.add_argument("--target-model", default=cfg.tts.cloud_model,
                    help=f"绑定的合成模型(默认跟 config 走: {cfg.tts.cloud_model})")
    ap.add_argument("--list", action="store_true", help="列出已注册的音色")
    ap.add_argument("--delete", metavar="VOICE_ID", help="删除某个音色")
    a = ap.parse_args()

    key_env = cfg.tts.cloud_api_key_env
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"❌ 没设环境变量 {key_env} —— `setx {key_env} sk-xxx` 后重开终端")

    if a.list:
        body = _call({"model": "voice-enrollment",
                      "input": {"action": "list_voice", "page_index": 0, "page_size": 30}}, key)
        vs = (body.get("output") or {}).get("voice_list") or []
        print(f"已注册 {len(vs)} 个音色:")
        for v in vs:
            print(f"  {v.get('voice_id')}   {v.get('status', '')}   {v.get('gmt_create', '')}")
        return 0

    if a.delete:
        _call({"model": "voice-enrollment",
               "input": {"action": "delete_voice", "voice_id": a.delete}}, key)
        print(f"✅ 已删除 {a.delete}")
        return 0

    if not a.wav:
        raise SystemExit("❌ 要注册就把参考音频路径给我,或者用 --list / --delete")

    _check_wav(a.wav)
    rec = enroll(a.wav, a.target_model, a.prefix, key)

    print("\n✅ 音色注册成功")
    print(f"   voice_id     : {rec['voice_id']}")
    print(f"   绑定模型     : {rec['target_model']}")
    print(f"   参考音频     : {rec['ref_audio']} ({rec['ref_secs']}s, sha256:{rec['ref_sha256']})")
    print("\n下一步:把 config.yaml 的 tts 段改成")
    print("   backend: cosyvoice")
    print(f"   cloud_model: {rec['target_model']}")
    print(f'   cloud_voice_id: "{rec["voice_id"]}"')
    print("\n然后试听:  python main.py tts \"你好,我是小mo\"")
    print("⚠️ 试听时如果打印了「降级说明」,说明云端没走通、用的是 edge 音色,别误判成克隆。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
