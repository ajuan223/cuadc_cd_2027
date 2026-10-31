# 本地视频窗口识别

在项目根目录运行（Python 环境需要 `opencv-python`、`numpy`、`onnxruntime`）：

```powershell
python photo/digit_recognition/app.py --source "photo/video/DJI_20260906165924_0026_D_stabilized.mp4" --infer-every 1
```

替换 `--source` 即可使用其他本地视频，路径有空格时保留引号。不要添加 `--headless`，该参数会关闭窗口显示。

- 框旁显示目标编号 `T`、当帧数字 `num`、跨帧投票数字 `vote`、检测分数 `det` 和数字识别置信度 `OCR`。
- 绿色 `OK` 表示当前数字通过 OCR 阈值；橙色 `CHECK` 表示本帧尚未通过，`vote` 仍可能保留此前结果。`OFF` 表示关闭了 OCR。
- 默认 `--target-proposal color` 的 `det` 为颜色和几何规则评分；使用 `--target-proposal yolo` 时才是检测模型输出的置信度。
- 画面显示推理耗时、帧号、视频时间和源视频 FPS。源 FPS 不代表实际推理速度，推理慢时播放会变慢。
- 空格暂停/继续，`Q`、`Esc` 或关闭窗口退出；可以拖动窗口边缘调整显示大小。
- `--infer-every 1` 逐帧检测，保证显示框与该帧对应；大于 1 时沿用已有的抽帧策略，中间帧重复上次识别画面。

保存带标注的视频：

```powershell
python photo/digit_recognition/app.py --source "photo/video/DJI_20260906165924_0026_D_stabilized.mp4" --infer-every 1 --output "photo/output/digit_preview.mp4"
```

只检查画框和数字区域提取时添加 `--no-ocr`，无需加载 OCR 模型。窗口需要带 GUI 的 OpenCV，`opencv-python-headless` 不支持弹窗。
