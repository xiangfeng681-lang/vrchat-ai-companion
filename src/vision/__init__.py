"""视觉 —— 让小mo 看得见 VRChat 里发生了什么。

模块划分:
  screen  抓 VRChat 窗口的画面(PrintWindow,被遮挡/异屏也能抓)
  detect  从画面里找名牌(自训练的 YOLO,导出成 ONNX 走 CPU)
"""
