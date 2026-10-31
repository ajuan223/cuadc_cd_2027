# V1 第二步：从检测框中裁剪靶标

这一步在第一步的基础上增加一个动作：

```text
原图 -> target_detector -> 检测框 -> NumPy 图像切片 -> 靶标小图
```

本步仍然不加载数字模型或图案模型。

## 原理

检测框给出左上角 `(x1, y1)` 和右下角 `(x2, y2)`：

```text
(x1, y1) ┌─────────┐
         │  靶标   │
         └─────────┘ (x2, y2)
```

OpenCV 读取的图片本质上是 NumPy 数组，因此裁剪只需要：

```python
crop = image[y1:y2, x1:x2].copy()
```

代码会先把浮点坐标转成安全的整数边界，防止越界。

## 运行

```powershell
.\.venv\Scripts\python.exe tools\step2_crop_target.py
```

指定其他图片：

```powershell
.\.venv\Scripts\python.exe tools\step2_crop_target.py --image path\to\image.jpg
```

## 输出

裁剪结果保存在：

```text
artifacts/step2_crops/target_001.jpg
```

终端会打印检测框、置信度和裁剪图尺寸。打开小图，确认其中只包含靶标及其紧邻区域。