好的，这是一份从零开始的完整学习记录，按**实际操作顺序**整理，你可以直接复制到今日学习文档中。

---

# nanoVLM 本地部署与推理 —— 完整操作记录

**日期**：2026年8月13日  
**目标**：从零开始，在本地 Windows 电脑上部署 nanoVLM 视觉语言模型，并成功运行推理。

**硬件环境**：
- 显卡：NVIDIA GeForce RTX 4070 Laptop GPU（8GB 显存）
- 系统：Windows
- CUDA 版本：12.8（已安装）

**项目参考**：
- nanoVLM 官方仓库：https://github.com/huggingface/nanoVLM
- 预训练模型：`lusxvr/nanoVLM-230M-8k`


## 一、准备工作：安装基础软件

### 1.1 安装 Python 3.12

- 去 python.org 下载 Python 3.12.x 安装包
- **关键操作**：安装时勾选“Add Python to PATH”
- 验证：命令行输入 `python --version`，显示版本号即成功

### 1.2 安装 Git

- 去 git-scm.com 下载 Git 安装包
- 全部保持默认选项完成安装
- 作用：从 GitHub 下载项目代码


## 二、下载项目代码

```bash
# 克隆 nanoVLM 仓库到本地
git clone https://github.com/huggingface/nanoVLM.git

# 进入项目文件夹
cd nanoVLM
```


## 三、创建并激活 Python 虚拟环境

```bash
# 创建虚拟环境（名为 venv）
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate
```

**激活成功后**，命令行前会出现 `(venv)` 字样，代表已进入独立环境。所有后续安装都在此环境下进行，不影响系统全局 Python。


## 四、安装核心依赖

### 4.1 安装 PyTorch（GPU 版，CUDA 12.8）

**遇到问题**：第一次用清华源默认安装了 CPU 版本，导致 `torch.cuda.is_available()` 返回 `False`。

**解决方案**：先卸载 CPU 版，再从官方源安装 GPU 版。

```bash
# 卸载 CPU 版
pip uninstall torch torchvision torchaudio

# 从南京大学镜像源安装 CUDA 12.8 版本
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128
```

**文件大小**：约 2.75 GB，下载耗时约 4 分钟。  
**验证**：`torch.cuda.is_available()` 返回 `True`。

> **经验**：国内用户可用 `mirrors.nju.edu.cn`（南京大学）或 `mirrors.tuna.tsinghua.edu.cn`（清华）镜像加速下载。

### 4.2 安装剩余依赖包

```bash
pip install numpy pillow datasets huggingface-hub transformers wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## 五、配置 Hugging Face 国内镜像源

Hugging Face 官网访问较慢，配置国内镜像站加速模型下载。

**方法**：在系统环境变量中设置：
- 变量名：`HF_ENDPOINT`
- 变量值：`https://hf-mirror.com`

**生效**：配置后需重新打开命令行窗口。


## 六、运行推理测试

```bash
python generate.py
```

**首次运行**：自动下载模型权重 `model.safetensors`（约 912 MB）。

### 6.1 遇到的错误及修复

#### 错误 1：`ModuleNotFoundError: No module named 'einops'`

**原因**：`einops` 是项目需要的张量操作库，之前未安装。  
**修复**：
```bash
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 错误 2：`RuntimeError: Could not infer dtype of tokenizers.Encoding`

**原因**：`apply_chat_template` 返回对象类型与新版 transformers 不兼容。  
**修复**：修改 `generate.py` 第 79-80 行：

```python
# 原代码
tokens = torch.tensor(encoded_prompt).to(device)

# 修改为
tokens = encoded_prompt.input_ids.to(device)
```

### 6.2 最终运行结果

```text
Using device: cuda
Loading weights from: lusxvr/nanoVLM-230M-8k
Resize to max side len: True

Input:
  What is this?

Output:
  >> Generation 1: This is a cat, specifically a tabby with distinct stripes...
  >> Generation 2: This image features a close-up of a tabby cat...
  >> Generation 3: This image features a domestic cat, likely a tabby...
  >> Generation 4: This image features a cat sitting on a patterned carpet...
  >> Generation 5: This image features a calico cat...
```

**说明**：模型默认识别 `assets/image.png`（一张猫的图片），并生成 5 条不同的描述。


## 七、自定义输入测试

### 7.1 换用自定义图片

```bash
# 基本用法
python generate.py --image "你的图片路径"

# 示例：识别熊猫图，只生成 1 条回答
python generate.py --image "panda.jpg" --generations 1
```

### 7.2 修改提问内容

```bash
python generate.py --image "panda.jpg" --prompt "What color is this panda?" --generations 1
```

**结论**：模型能根据问题调整回答内容，验证了 VLM 的指令跟随能力。


## 八、代码理解：parse_args() 的作用

```python
parser.add_argument("--image", type=str, default="assets/image.png")
```
- 通过命令行参数灵活指定输入图片，无需修改代码
- `default` 为默认值，不指定时自动使用

```python
parser.add_argument("--prompt", type=str, default="What is this?")
```
- 自定义对图片的提问内容

```python
parser.add_argument("--generations", type=int, default=5)
```
- 控制生成回答的数量


## 九、关于“输出确定性”的发现

**现象**：同一张图、同一个问题，多次运行输出完全一致。  
**原因**：
1. `torch.manual_seed(0)` 固定了随机种子
2. 默认使用“贪心解码”而非“采样解码”

**对项目意义**：确定性的输出有利于消融实验的公平对比（控制变量）。


## 十、学到的东西

1. **Python 虚拟环境**：隔离项目依赖，避免版本冲突
2. **PyTorch GPU 版安装**：需指定 `--index-url` 明确 CUDA 版本，否则可能装成 CPU 版
3. **国内镜像源配置**：PyPI 用清华/中科大，HuggingFace 用 hf-mirror.com
4. **开源项目协作**：fork/clone、本地修改、版权合规（保留 LICENSE，注明出处）
5. **VLM 推理流程**：图片 → ViT → Projector → LLM → 文本输出
6. **消融实验要求**：固定随机种子，确保结果可复现


## 十一、后续计划

1. 理解 WASD 频谱数据集（.npy 格式）
2. 将频谱图接入模型的图片输入通道
3. 设计 6 类频谱 VQA 任务
4. 开展 4 组 Projector 结构的消融实验


**学习笔记结束** 🎉


**补充建议**：你可以把这份文档同步到你刚才创建的 GitHub 仓库中，保存为 `LEARNING_LOG.md`，这就是第一份正式的学习记录。后续每天的操作都可以按此格式追加，形成完整的时间轴。