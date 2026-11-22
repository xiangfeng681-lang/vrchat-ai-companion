"""跳舞模式开关 —— 跳舞时把会打架、会干扰的东西全停掉。

## 为什么必须停

🔴 **`pose_server` 和 AnyaDance 播舞推的是同一个口(39570)。**
两个一起推 → 驱动收到交错的帧 → **模型抖动**。所以跳舞前必须停 pose_server。

另外两样是**用户要求**的:

- **视觉**(名牌定位 / 眼睛)—— 跳舞时不需要,而且它一直在读画面、发转向指令
- **语音**(耳朵 / 大脑 / 嗓子)—— 跳舞时不该还在听人说话、突然插一句

## 用法

    python tools/dance_mode.py on      # 进入跳舞模式(停掉上面三样)
    python tools/dance_mode.py off     # 退出(把语音和动作拉回来)
    python tools/dance_mode.py         # 看看现在是什么状态

⚠️ **`on` 不会停 GPT-SoVITS** —— 它只是个常驻的 TTS 服务,不推姿态也不抢驱动,
   留着下次退出跳舞模式时不用重新冷启动(省 40~60 秒)。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def _ps(cmd: str) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "").strip()


# 用命令行特征匹配,而不是按 PID —— 每次重启 PID 都变。
_QUERY = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object {{ $_.CommandLine -match '{pat}' }}")


def running(pat: str) -> int:
    out = _ps(_QUERY.format(pat=pat) + " | Measure-Object | Select-Object -ExpandProperty Count")
    try:
        return int(out.splitlines()[-1])
    except Exception:
        return 0


def kill(pat: str, label: str) -> None:
    n = running(pat)
    if not n:
        print(f"  {label}: 本来就没在跑")
        return
    _ps(_QUERY.format(pat=pat) + " | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
    print(f"  {label}: 已停")


def start(script: str, label: str) -> None:
    subprocess.Popen(
        [PY, "-u", str(ROOT / script)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    print(f"  {label}: 已拉起")


def status() -> None:
    print("现在:")
    print(f"  pose_server(动作/转向) : {'在跑' if running('pose_server') else '没跑'}")
    print(f"  pipeline  (耳朵/眼睛)  : {'在跑' if running('main\\.py') else '没跑'}")
    print(f"  GPT-SoVITS(嗓子服务)   : {'在跑' if running('api_v2') else '没跑'}")


def main() -> int:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if mode == "on":
        print("进入跳舞模式(AnyaDance 独占驱动):")
        kill("pose_server", "pose_server")   # 🔴 必须停,否则和播舞打架
        kill("main\\.py", "语音管线")         # 用户要求:跳舞时关视觉和语音
        print()
        print("现在去 AnyaDance 里:**关掉 UI 之外什么都别开**,直接用 Dance (MMD) 播舞。")
        print("⚠️ 播舞时**不要**打开 pose_server —— 会抖。")
    elif mode == "off":
        print("退出跳舞模式,把服务拉回来:")
        start("tools/pose_server.py", "pose_server")
        print("  (语音管线稍微慢一点,等 3 秒)")
        import time
        time.sleep(3)
        # 管线要带 DASHSCOPE key(云端 ASR/LLM/视觉都靠它)
        import os
        env = dict(os.environ)
        key = _ps("[Environment]::GetEnvironmentVariable('DASHSCOPE_API_KEY','User')")
        if key:
            env["DASHSCOPE_API_KEY"] = key
        subprocess.Popen([PY, "-u", "main.py", "run"], cwd=str(ROOT), env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                         | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        print("  pipeline: 已拉起(约 20 秒后能用)")
    else:
        status()
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
