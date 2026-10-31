# V1 第一步：单张图片靶标检测

这一步只回答一个问题：

> 靶标模型能不能在图片里找到靶标？

程序会：

1. 读取 `test.jpg`；
2. 加载 `digit_recognition/models/target_detector.onnx`；
3. 输出所有检测到的靶标框和置信度；
4. 把框画到图片上并保存到 `artifacts/step1_detected.jpg`。

这一步不加载数字模型，也不加载图案模型。

## 安装依赖

在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-step1.txt
```

## 运行

```powershell
.\.venv\Scripts\python.exe tools\step1_detect_image.py
```

也可以指定自己的图片：

```powershell
.\.venv\Scripts\python.exe tools\step1_detect_image.py --image path\to\image.jpg
```

## 查看结果

终端会打印：

```text
targets=1
target_1: box=(x1, y1, x2, y2) confidence=0.9xxxx
output=...\artifacts\step1_detected.jpg
```

打开 `artifacts/step1_detected.jpg`，查看靶标外是否出现绿色检测框。

## 可以尝试

提高置信度阈值：

```powershell
.\.venv\Scripts\python.exe tools\step1_detect_image.py --confidence 0.7
```

置信度阈值越高，模型越严格，通常会减少误检，但也可能漏掉目标。

## 本次基线结果

使用 `test.jpg` 和默认阈值 `0.6` 时，检测结果为一个真实靶标：

```text
box=(213.7, 162.5, 401.4, 277.1) confidence=0.7434
```

如果把阈值降低到 `0.25`，模型还会在图片右侧的道路边缘产生一个低置信度误检。这说明置信度阈值的作用是控制“宁可漏检，还是宁可误检”，后续需要根据实际飞行数据调整，不能只凭一张图片确定。
