# V1 第三步：识别数字靶

这一步把前面几个步骤串起来：

```text
靶标检测框
→ 向外扩 15% 形成完整靶标裁剪图
→ 提取并旋正数字面板
→ digit_model 识别两位数字
```

图案模型不会在这一步加载。

## 为什么不能直接用第二步的紧框

第二步使用检测框的精确边界，适合学习“裁剪”本身。但数字模型还需要靶标白色边缘、箭头方向和少量背景来判断朝向。如果框切得太紧，几何提取可能把整块靶标误当成数字面板，导致 OCR 识别失败。

因此第三步使用 `padded_crop(..., padding=0.15)`，在检测框四周各扩出一定比例。

## 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-step3.txt
```

## 运行

```powershell
.\.venv\Scripts\python.exe tools\step3_recognize_digit.py
```

默认使用：

```text
testdata/digit_17_frame.jpg
```

这是从实际飞采数据中挑选的一帧，目标数字为 `17`，便于重复验证。

指定其他图片：

```powershell
.\.venv\Scripts\python.exe tools\step3_recognize_digit.py --image path\to\image.jpg
```

## 输出

中间图片和数字条带保存在：

```text
artifacts/step3_digit/
```

终端会打印：

```text
target_1: detection=... crop=... geometry=... strips=1 rejection=None
result: number=17 score=... accepted=True raw=17
```

## 已知情况

仓库根目录的 `test.jpg` 能被靶标模型检测到，但现有数字几何和 OCR 模型无法稳定识别其中的 `17`。因此第三步选用实际飞采帧 `digit_17_frame.jpg` 作为可重复成功样例。这个问题保留为后续模型与几何算法优化项，不能通过在测试脚本里硬编码结果来绕过。
## 本次运行结果

`digit_17_frame.jpg` 中，靶标检测模型返回了 4 个候选框：

1. 白色车辆：数字几何拒绝；
2. 红白数字靶：几何提取成功，OCR 输出 `17`；
3. 车辆附近的误检：数字几何拒绝；
4. 大范围误检：数字几何拒绝。

最终有效结果为：

```text
number=17
score=0.9960
accepted=True
raw=17
```

因此本脚本默认最多检查 8 个检测框，并让数字几何承担二次筛选，而不是假定置信度最高的框一定是数字靶。

这次运行同时验证了两个已知问题：

- 靶标检测模型会把车辆误认为 `target`，需要继续训练；
- 数字几何筛选能够过滤掉大部分非数字靶误检，但不能替代检测模型本身的改进。
