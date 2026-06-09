# animal_detect

基于 TensorFlow 2.2 / `tf.keras` 的动物图片分类示例，支持命令行预测和 PyQt5 图形界面预测。模型可识别 5 类动物：

| 标签 | 类别 |
| --- | --- |
| 0 | bird |
| 1 | butterfly |
| 2 | cat |
| 3 | dog |
| 4 | tiger |

## 环境

本项目复用本机已有的 `tf22` conda 环境，不安装独立 `keras` 包。`tf22` 中的 `tensorflow-gpu==2.2.0` 已经能识别 CUDA 10.1 和本机 GPU，因此预测代码使用 TensorFlow 自带的 `tf.keras`。

建议用 `conda run` 执行命令，避免直接调用环境内 `python.exe` 时缺少 DLL/SSL 路径：

```powershell
F:\anaconda\Scripts\conda.exe run -n tf22 python -m pip install -r requirements.txt --trusted-host pypi.mirrors.ustc.edu.cn
```

如果已经在终端中激活了环境，也可以直接运行：

```powershell
conda activate tf22
python -m pip install -r requirements.txt --trusted-host pypi.mirrors.ustc.edu.cn
```

## 命令行预测

```powershell
F:\anaconda\Scripts\conda.exe run -n tf22 python predict.py --image samples\bird_0_1.jpg
```

输出中会包含预测类别和置信度，例如：

```text
result: bird
confidence: 99.99%
```

## 图形界面预测

不传 `--image` 参数时会启动 PyQt5 图形界面：

```powershell
F:\anaconda\Scripts\conda.exe run -n tf22 python predict.py
```

点击“打开图片”选择本地图片后，界面会显示图片、预测类别和置信度。

## 模型与数据

- `animal.h5` 随仓库提交，预测时默认从当前目录加载。
- `samples/` 只保留每类少量示例图片，用于快速验证。
- 完整图片数据集目录 `myPic/`、预处理后数据集目录 `pic_object/`、训练 checkpoint 目录 `checkpoint1/` 不提交到 Git。

## 训练脚本

`mobilev2.py` 是训练脚本，会读取 `pic_object/` 数据集并保存 `animal.h5`。训练依赖完整本地数据集，仓库中不包含完整数据集。

训练脚本里的 MobileNetV2 保留了 `weights='imagenet'`，首次训练会自动下载 ImageNet 预训练权重。预训练权重对小数据集更友好，因此不要改成 `weights=None`，除非明确想从零开始训练。

请用 `conda run` 或已经激活的 `tf22` 环境启动训练，不要直接调用 `C:\Users\weili\.conda\envs\tf22\python.exe`。直接调用环境内的 `python.exe` 可能没有加载 conda 的 DLL/SSL 路径，导致下载权重时报错：

```text
unknown url type: https
```

训练前可以先验证 SSL 是否正常：

```powershell
F:\anaconda\Scripts\conda.exe run -n tf22 python -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

正常时会输出类似 `OpenSSL 1.1.1w`。然后启动训练：

```powershell
F:\anaconda\Scripts\conda.exe run -n tf22 python mobilev2.py
```

第一次训练时，Keras 会把 MobileNetV2 权重下载到：

```text
C:\Users\weili\.keras\models\
```

### VS Code 点击训练

仓库已提供 `.vscode/tasks.json`。在 VS Code 中可以打开命令面板执行 `Terminal: Run Build Task`，或使用默认构建快捷键，选择/运行 `训练 MobileNetV2`。这个任务会通过 `F:\anaconda\Scripts\conda.exe run -n tf22 python mobilev2.py` 启动，避免 SSL DLL 没加载的问题。
