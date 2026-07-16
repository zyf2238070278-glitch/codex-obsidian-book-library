# macOS Apple Silicon 真离线全量安装包设计

## 目标

为 Codex Obsidian 本地书库生成一个可转交他人的 macOS Apple Silicon ZIP。接收者在没有 Homebrew、Python、uv、pip、Xcode Command Line Tools、预热缓存或网络连接的情况下，解压后运行 `install-macos.command` 即可完成本地运行环境安装。安装成功后，用户只需重启 Codex、打开并信任整个项目目录，即可检查书库状态、导入书籍、检索原文及运行本地 OCR。

“离线”只覆盖安装过程、语义检索、索引和 OCR。Codex 对话及模型回答仍可能需要 Codex 服务联网。发布包不得包含维护者或用户的原书、数据库、笔记、OCR checkpoint、本机配置、凭据、缓存或绝对路径。

## 兼容范围

- 处理器：全部 Apple Silicon M 系列，发行载荷只允许 `arm64` 或包含 `arm64` slice 的 `universal2` 二进制。
- 系统：最低 macOS 14；当前发布机必须在 macOS 26 上完成真实离线集成测试。
- Python：固定 CPython 3.12.11 的 `python-build-standalone` macOS arm64 资产，不允许在目标机选择其他补丁版本。
- 安装目录：支持绝对路径中含空格、中文和较长目录名。
- Gatekeeper：在没有 Developer ID 证书和公证的前提下，首次运行可能需要用户右键选择“打开”或在“隐私与安全性”中确认；安装器不得清除 quarantine，也不得降低系统安全设置。

当前锁定的 Torch、NumPy、SciPy 和 ONNX Runtime 等原生 wheel 中存在最低部署目标为 macOS 14 的 Mach-O，因此不承诺 macOS 13 或更早版本。

## 方案选择

采用“固定 Python 本地镜像 + 平台专用 wheelhouse + 目标目录内离线建环境”。不直接复制发布机现有 `.venv`，也不使用 PyInstaller/Nuitka 冻结整个服务。

普通 `.venv` 包含基础解释器绝对符号链接、`pyvenv.cfg`、入口脚本 shebang、激活脚本及 `.pyc` 构建路径，不能可靠搬迁到另一台 Mac。目标机在自己的最终目录中创建 `.venv`，可以从根源上避免维护者路径泄漏和重定位失败。

## 发布包结构

新版本使用独立的离线发行元数据和名称，避免与当前会联网安装的 `v0.2.0-beta.1` 混淆。首个候选版本为 `v0.3.0-beta.1`：

```text
codex-obsidian-book-library-v0.3.0-beta.1-macos-arm64-offline/
├── install-macos.command
├── bin/
│   ├── uv
│   └── book-vision-ocr
├── offline/
│   ├── python-mirror/
│   │   └── 20251007/
│   │       └── cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz
│   ├── wheelhouse/
│   │   └── *.whl
│   ├── requirements-macos-arm64-py312.txt
│   ├── python-manifest.json
│   └── wheelhouse-manifest.json
├── data/models/
│   └── multilingual-e5-small 固定快照
├── book_agent/
├── installer/
├── docs/
├── third_party/
├── RELEASE-MANIFEST.json
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
└── uv.lock
```

CPython 资产固定为：

```text
文件：cpython-3.12.11+20251007-aarch64-apple-darwin-install_only_stripped.tar.gz
SHA-256：407fa242942a7ba5d91899abc562fc9897f7a0376f8d2060285e8c0560323f19
```

wheelhouse 是 `uv.lock` 在 CPython 3.12、macOS arm64、启用 `semantic` 和 `ocr` extra、排除开发依赖后的完整闭包。每个依赖必须被固定为一个 wheel；不允许把 sdist 交给目标机编译。`antlr4-python3-runtime==4.9.3` 没有上游 wheel，发布构建阶段必须在隔离环境中预构建 wheel，再将其哈希纳入清单。

## 构建流程

`scripts/build_macos_release.py` 扩展为离线发行构建器，并继续使用临时 staging、精确 allowlist、确定性 ZIP 时间戳、原子发布和最终重新打开验证。

构建器接收并验证以下可信输入：

1. 固定语义模型快照及现有模型清单。
2. 固定版本、SHA-256 和 arm64 架构的 `uv`。
3. 固定版本、schema、arm64 架构及签名状态的 Apple Vision helper。
4. 固定 CPython 资产及 `python-manifest.json`。
5. 与导出的 hash-locked requirements 完全一致的 wheelhouse 及 `wheelhouse-manifest.json`。

构建阶段执行以下检查：

- requirements 与 `uv.lock + semantic + ocr` 的目标平台闭包一致，wheelhouse 无缺失、无多余文件、无 sdist。
- 每个 Python 资产和 wheel 的文件名、大小与 SHA-256 与可信清单一致。
- wheel tag 适用于 CPython 3.12 macOS arm64，原生二进制只含 arm64/universal2。
- Mach-O 的 load commands 不依赖维护者目录、Homebrew 或包外非系统动态库；最低系统不高于声明的 macOS 14 基线。
- staging 和 ZIP 不含 `.git`、`.codex/config.toml`、`.venv`、用户 Vault、书籍、数据库、笔记、密钥、维护者用户名或项目绝对路径。
- 所有第一方文本继续执行通用绝对路径和凭据扫描。第三方固定二进制资产允许包含上游构建路径，但必须精确匹配可信哈希，并仍扫描维护者专属路径与凭据模式。
- `RELEASE-MANIFEST.json` 覆盖 ZIP 中每一个载荷文件的路径、大小、SHA-256 和模式。
- 两次相同输入的构建产生相同 ZIP 哈希。

发布包保留原始 wheel，不剥离其中的 `.dist-info/licenses`。`THIRD_PARTY_NOTICES.md` 和生成的依赖清单必须补充 CPython/PSF、python-build-standalone、uv、模型及所有随包 Python 依赖的再分发信息。

## 安装流程

### 启动阶段

`install-macos.command` 不再调用系统 `python3`，也不再使用可能联网的 `uv run --python 3.12`。它只调用包内 `bin/uv`：

1. 使用 `/usr/bin/uname` 和 `/usr/bin/sw_vers` 拒绝非 arm64 或低于 macOS 14 的系统。
2. 确认包内 `uv`、Python 资产、requirements 和清单均为普通文件，拒绝符号链接或缺失项。
3. 创建项目内事务目录 `.offline-install-stage-*/`，其中预先使用最终兄弟目录名 `.runtime/` 与 `.venv/`；同时创建空的事务内 uv cache，禁止继承用户缓存。
4. 使用 `file://` Python mirror 和固定版本，把 CPython 安装进事务目录的 `.runtime/`；不创建用户级命令、不修改 `PATH`、不查询网络 registry。
5. 用新安装的包内 Python 启动 `installer/install_macos.py`。

所有工具路径使用绝对路径或包内路径。安装在 `PATH` 为空时仍必须成功。

### 环境阶段

安装器在目标项目目录内执行：

```text
bin/uv venv --relocatable --python <事务目录/.runtime中的固定Python> <事务目录/.venv>

bin/uv pip sync \
  --python <事务目录/.venv>/bin/python \
  --offline \
  --no-index \
  --find-links <包内wheelhouse> \
  --require-hashes \
  --no-build \
  --no-python-downloads \
  --link-mode copy \
  <包内requirements>
```

安装器显式设置空的 `UV_CACHE_DIR`，并禁止从用户 site-packages 读取包。环境中的文件必须复制进项目，不得符号链接到发布包外或用户缓存。

### 自检与发布阶段

事务目录中的 runtime 和 venv 在发布前必须完成：

- Python 恰为 CPython 3.12 arm64，解释器与标准库路径均位于项目内。
- `uv pip check` 通过。
- 成功导入 `mcp`、`numpy`、`fitz`、`torch`、`sentence_transformers`、`onnxruntime`、`rapidocr`、`cv2` 和书库服务模块。
- 固定 E5 模型在离线变量启用时产生 384 维有限向量。
- RapidOCR 的三个 ONNX 模型存在且哈希正确，并完成合成小图烟测。
- Apple Vision helper 通过 Mach-O、arm64、codesign 和 capabilities 校验。
- MCP 服务可启动并返回 `tools/list` 与 `library_status`。

事务目录内部始终保持 `.runtime` 与 `.venv` 的最终相对布局，使 relocatable venv 的相对解释器引用在发布前后保持一致。预发布验证通过后，安装器先把现有 `.runtime`/`.venv` 重命名为恢复备份，再把事务目录内的两个新目录移动到项目根。它必须从最终路径重新执行 Python、关键 imports 和 `uv pip check`；只有最终路径复验通过，才删除恢复备份、创建默认 Vault 与运行目录，并原子写入 `.codex/config.toml`。任何移动或复验失败都恢复旧目录。配置中的 Python 指向项目内 `.venv/bin/python`，且保留 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONNOUSERSITE=1`。

任何失败都必须退出非零、显示明确中文错误、保留旧的可用环境与配置，并清除未发布的临时目录。重复运行安装器必须幂等；当联合指纹与全部自检一致时可以复用环境，否则在临时目录重建并原子替换。

## 数据与移动边界

发行 ZIP 只含程序与第三方运行载荷，不含任何书籍或用户数据。首次安装生成空数据库和空白 Vault 是正确行为，用户随后通过 Codex 导入自己的书。

`.codex/config.toml` 会写入目标电脑的绝对路径，因此项目移动后必须重新运行安装器。已有数据库也保存原书和解析文件的绝对路径；在另行实现数据库路径迁移前，文档必须要求用户安装并导入书籍后不要随意移动项目目录。离线发行本身不扩大为数据库迁移项目。

## 文档与用户体验

README、安装说明、常见问题、隐私说明和 Word 指南统一改为：

- 解压 ZIP，双击 `install-macos.command`，等待“安装完成”。
- 安装过程中不下载 Python、Python 包、语义模型或 OCR 模型，不需要手工配置环境。
- 安装后完全退出并重启 Codex，打开并信任整个项目目录，再说“检查书库状态”。
- 若 Gatekeeper 拦截，只允许右键“打开”或系统“隐私与安全性”的官方确认入口。
- 安装需要约 1.8 GB，建议预留 3–5 GB；最终数字以生成包实测为准。
- 本地书库与 OCR 可离线，Codex 对话仍可能联网。

`install-from-github.command` 只有在新的离线 ZIP 上传为 GitHub Release、校验文件可用且下载流程测试通过后才切换到 `v0.3.0-beta.1`。本地候选包不得让公开一行安装命令提前指向尚未发布的资产。

## 测试策略

### 单元与构建测试

- 先写失败测试，再实现构建器和安装器变化。
- 构建器测试覆盖 Python/wheel 清单、闭包一致性、缺失/多余/篡改资产、sdist、错误架构、过高部署目标、危险路径、秘密、本机路径、权限、确定性和原子回滚。
- launcher 测试覆盖空 `PATH`、无系统 Python、项目路径含中文与空格、错误 CPU/系统版本、子进程失败传播及禁止联网参数。
- installer 测试覆盖纯本地 venv 创建命令、hash-locked wheel 安装、解释器/import/模型/helper/MCP 自检、失败不发布配置、幂等重装和旧环境回滚。
- 文档契约测试拒绝“首次需要联网安装依赖”等旧说明，并要求明确安装离线与 Codex 对话边界。

### 当前 Mac 真实离线集成门

候选 ZIP 解压到新的中文加空格路径，使用全新临时 `HOME`、空 `UV_CACHE_DIR/XDG_CACHE_HOME/PIP_CACHE_DIR`、空 `PATH`，并由 macOS sandbox 明确拒绝网络访问。连续安装两次都必须退出 0。随后执行关键 imports、384 维 embedding、Vision/RapidOCR 烟测、MCP 状态、TXT 导入、检索和段落获取。

候选包还必须通过 ZIP CRC、清单、SHA-256、隐私扫描、Mach-O 架构/load command/部署目标和 codesign 审计。

### 外部兼容门

“在所有 M 系列和全部支持系统上验证”不能仅由一台发布机证明。正式公开发布前至少需要：

- 一台或 VM 的干净 macOS 14 Apple Silicon 环境。
- 一台干净 macOS 26 Apple Silicon 环境。
- 无 Homebrew/Python/uv/pip/Xcode CLT，虚拟机层断开网卡。
- 通过 Finder 解压带 quarantine 的 ZIP，并走正常 Gatekeeper 首次打开流程。

在无法取得最低系统测试机时，可以交付标记为候选版的 ZIP及当前 macOS 26 的验证报告，但不得声称已经实机覆盖 macOS 14。

## 完成标准

以下条件全部满足才称为“真离线全量 ZIP”：

1. ZIP 内含固定 Python 资产、完整 wheelhouse、语义模型、OCR 运行时、安装器及源码。
2. 全新空缓存、空 `PATH`、网络硬阻断环境可以完成安装。
3. 安装期间没有依赖下载或编译行为，不要求用户手工安装环境。
4. 解压到不同中文/空格路径后仍能创建可运行的本地 `.venv`，配置不含发布机路径。
5. 语义检索、MCP、Apple Vision 和 RapidOCR 烟测通过。
6. 包内没有书籍、数据库、笔记、个人配置、凭据或发布机绝对路径。
7. SHA-256、完整载荷清单、第三方许可和用户文档随包交付。
