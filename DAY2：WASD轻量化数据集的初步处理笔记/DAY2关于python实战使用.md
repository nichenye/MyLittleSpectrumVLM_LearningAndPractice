## 📚 从零到一：Python环境配置与项目运行全记录

恭喜你成功跑通了代码！这不仅仅是运行了一个脚本，而是你亲手搭建了一个完整的Python开发环境，并让一个VLM模型在你的电脑上完成了推理。这个过程几乎是所有Python开发者的“必修课”，你现在已经掌握了最核心的技能。


### 第一阶段：设置Python解释器（最关键的环节）

**问题**：你装了Python，但VS Code里运行代码时找不到torch、matplotlib等包。

**根源**：电脑里可能有多个Python——系统全局的、虚拟环境的、Jupyter自带的。VS Code默认用了“全局”那个，而你的包装在“虚拟环境”里。

**解决方案**：
1. **找到虚拟环境里的python.exe**：通常在你项目目录的 `venv/Scripts/python.exe`
2. **在VS Code里按 `Ctrl+Shift+P` → 输入“Python: Select Interpreter”**
3. **选择路径指向 `venv/Scripts/python.exe` 的那个解释器**

> 💡 **核心经验**：你以后任何时候想确认“我在用哪个Python”，就在终端输入：
> ```bash
> where python   # Windows
> which python   # Mac/Linux
> ```
> 第一行结果就是当前默认的Python。如果看到不是虚拟环境的路径，说明解释器没选对。


### 第二阶段：用正确的pip安装包

**问题**：你在虚拟环境里运行 `pip install torch`，结果装到了系统全局Python（`d:\dev\python\...`）里，虚拟环境里还是找不到。

**根源**：终端里的 `pip` 命令默认用的是系统全局的pip，而不是虚拟环境的。

**解决方案**：
- **方法一（最可靠）**：直接用虚拟环境的python去调pip
  ```bash
  venv\Scripts\python.exe -m pip install torch
  ```
  这样安装的包会100%进入虚拟环境。
- **方法二**：先激活虚拟环境，再用pip
  ```bash
  venv\Scripts\activate      # 激活
  pip install torch           # 此时pip已指向虚拟环境的pip
  ```

> 💡 **核心经验**：当你看到“Requirement already satisfied”后面跟着的路径是 `d:\dev\python\...`，说明包装到了全局。只有看到 `...venv\...` 路径才说明装进了虚拟环境。这条判断方法可以帮你解决95%的环境问题。


### 第三阶段：在VS Code中一次性配置好

要在VS Code里轻松运行代码，你需要做三件事：

1. **选对解释器**：按 `Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `...venv/Scripts/python.exe`
2. **装对依赖**：在VS Code终端（自动使用虚拟环境）执行 `pip install ...`
3. **直接点运行**：配置好后，打开.py文件，直接点右上角的▶按钮就能运行

> 💡 **核心经验**：以后在任何项目中，这三步都是标准动作。你现在的项目文件夹里有一个 `venv` 文件夹，它就是你的“专属Python环境”，所有包都装在里面，和系统全局Python完全隔离。


### 附带的经验：Jupyter vs 纯Python脚本

你之前用Jupyter能跑通但VS Code跑不通，是因为Jupyter Notebook有自己的内核选择器。现在你学会了用VS Code，以后可以统一用VS Code管理所有Python代码，不用两个环境来回切换了。


### 最终的心法总结

所有环境问题归根结底只有一个问题——**“我到底在用哪个python.exe？”**

| 问题表象 | 检查方法 | 解决方法 |
|---|---|---|
| ModuleNotFoundError | 在终端执行 `where python` | 在VS Code里切换到正确的解释器 |
| 包装到了全局而非虚拟环境 | 看pip安装日志里的路径是哪里 | 用 `venv/Scripts/python.exe -m pip install` |
| VS Code终端里找不到虚拟环境 | 终端前有没有 `(venv)` 字样 | 手动执行 `venv\Scripts\activate` |


