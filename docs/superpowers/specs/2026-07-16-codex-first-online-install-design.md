# Codex 驱动的 Git 在线安装设计

## 目标

朋友从 GitHub 克隆项目后，在 Codex 中打开并信任整个项目，只需说：

> 请安装并检查这个书库。

Codex 按项目内指令运行唯一安装入口，联网准备运行环境、语义模型和
OCR 模型。朋友不需要理解或手动输入 Python、uv、pip、MCP、模型下载或
OCR 配置命令。安装完成后只需要完全退出并重启一次 Codex。

本设计替代此前的完全离线 ZIP 方案。此前的离线发布计划停止执行，不是
本方案的交付要求。

## 支持范围

- Apple Silicon（M 系列）Mac。
- 按用户要求，目标系统为 macOS 16 或更高版本。
- 安装阶段允许访问 GitHub、Python 包索引和 Hugging Face。
- Codex 本身保持正常联网。
- Git 仓库不包含用户书籍、SQLite 数据库、解析文本、AI 笔记或个人绝对路径。
- 日常检索和 OCR 仍在本机运行；只有进入 Codex 对话的少量检索证据参与在线回答。

## 用户流程

1. 朋友执行项目 README 中的 `git clone` 命令。
2. 在 Codex 中打开并信任克隆得到的整个项目目录。
3. 新建任务并说“请安装并检查这个书库”。
4. Codex 读取 `AGENTS.md` 的安装规则，运行根目录唯一安装入口。
5. 安装入口完成下载、配置和自检，明确输出“安装完成”或可操作的失败原因。
6. 朋友完全退出并重启一次 Codex，再次打开同一项目。
7. 新建任务并说“检查书库状态”；状态正常后即可导入书籍、检索和按需启动 OCR。

## 架构

### 1. Codex 安装契约

`AGENTS.md` 增加独立的“首次安装与修复”规则：

- 用户提出安装、初始化、修复环境或克隆后检查时，Codex 运行唯一安装入口。
- Codex 不自行拼接另一套 Python/pip 命令，也不猜测模型目录。
- 安装失败时，Codex 读取退出码和错误信息，修复明确问题后重试同一入口。
- 安装成功后，Codex 告知用户必须完整重启一次，不能假装 MCP 已立即加载。
- 重启后使用真实 `library_status` 验证，不把“脚本退出 0”当作书库工具已经可用。

### 2. 唯一在线安装入口

保留 `install-macos.command` 作为人和 Codex 共用的唯一入口，并让它能够从普通
Git 克隆状态启动。Git 直接携带固定的 arm64 uv 和小型 Vision helper，因此不依赖
预装 Python、uv、Xcode 或 Command Line Tools：

1. 校验 Darwin、arm64 和最低系统版本。
2. 校验项目内 `bin/uv` 的版本、arm64 架构和固定 SHA-256；缺失或损坏时明确要求
   Codex 从 Git 恢复该文件，不使用 PATH 中的其他 uv。
3. 使用项目内 uv 联网获取固定 Python 3.12，并在项目内创建 `.venv`。
4. 运行现有 Python 安装器完成剩余步骤。

重复运行必须幂等：已校验的 uv、Python、依赖和模型可复用，缺失或损坏的组件
重新准备，不删除已有书籍、数据库、Vault 或笔记。

### 3. Python 与依赖

安装器使用锁文件执行等价于下面范围的同步：

- 核心解析与 MCP 依赖；
- `semantic` 语义检索依赖；
- `ocr` RapidOCR 与 ONNX Runtime 依赖；
- 固定 Python 3.12；
- 项目专用 `.venv`，不修改朋友的系统 Python。

安装过程允许联网，但版本必须继续由 `uv.lock` 固定。失败信息区分网络失败、磁盘
不足、架构不支持、哈希错误和包安装失败，方便 Codex继续处理。

### 4. 语义模型与 OCR

- 语义模型固定为 `intfloat/multilingual-e5-small` revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`。
- 安装器把模型下载到项目 `data/models`，并使用现有
  `distribution/model-manifest.json` 验证必需文件、大小和 SHA-256。
- RapidOCR 的三个 ONNX 模型从锁定版本的已安装 wheel 复制到
  `data/ocr-models/rapidocr`，沿用现有安装器校验。
- 约 201 KB 的 Apple Vision arm64 helper 直接纳入 Git，避免朋友安装 Xcode 或
  Command Line Tools；安装器继续验证 Mach-O 架构、代码签名和 capabilities。
- `.gitignore` 只为这个固定 helper 增加精确例外，仍忽略其他本机运行数据。

### 5. 配置与自检

安装器根据朋友的实际克隆路径生成 `.codex/config.toml`，其中 MCP 命令固定指向
项目 `.venv/bin/python`，工作目录和环境变量指向当前项目与 Vault。

成功前必须完成以下自检：

- Python 3.12 和锁定依赖可导入；
- 语义模型能从 `data/models` 本地加载并生成 384 维向量；
- RapidOCR 三个模型存在；
- Apple Vision helper capabilities 正常；
- MCP 服务能够启动并返回工具列表；
- 新建空书库时 `library_status` 能正常返回。

自检不自动导入书籍，也不自动启动 OCR。

## 失败与恢复

- 下载文件先写入临时文件，哈希通过后再替换正式文件。
- 任何一步失败都返回非零退出码和中文错误；不写入指向不存在环境的 Codex 配置。
- 网络中断后重复运行同一入口即可继续，已完成且通过校验的部分不重复下载。
- 如果项目被移动，重新运行同一入口即可刷新绝对路径配置。
- 已存在的数据库、Vault、原书和笔记始终保留。

## 验证范围

实现使用测试驱动，重点验证：

1. 普通 Git 克隆中确实包含安装入口、固定 uv、Codex 指令和 Vision helper。
2. 没有系统 Python、没有 uv 时，安装入口会下载并校验项目内 uv。
3. 安装器会同步全部运行依赖、下载并验证固定语义模型、准备 OCR 模型。
4. 二次运行不会破坏已有数据。
5. 在新的临时克隆路径执行一次真实在线安装和自检。
6. README 给朋友的操作只保留“克隆、Codex 打开并说安装、重启、检查状态”。

本方案不再要求完全断网安装、离线 wheelhouse、内置 Python 或大型离线 ZIP。
