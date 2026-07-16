# 当前书库原地升级与可分发安装包设计

## 背景与根因

当前 Codex 项目配置仍从 `<PROJECT_ROOT>` 启动书库服务。双引擎 OCR 新版代码位于 `codex/multi-engine-ocr` 工作分支，因此当前会话实际运行的仍是旧版单引擎实现。旧版会把页面像素上限或 Vision 边界框异常直接记为整本任务失败，没有进入 RapidOCR 回退。

## 目标

1. 在不移动书籍、不重建数据库、不改变 Obsidian Vault 的前提下，将当前活动项目原地升级到双引擎 OCR。
2. 保留用户对 `AGENTS.md` 的修改和未跟踪的本地辅助脚本。
3. 安装 Apple Vision schema 2 helper、RapidOCR 运行依赖及固定模型文件。
4. 升级后只验证服务和 OCR 引擎，不自动重新启动任何失败或待处理书籍的 OCR。
5. 重新生成并发布可直接交给朋友使用的 Apple Silicon Mac 全量 ZIP，内含 Word 安装说明和使用说明。

## 方案

### 当前活动项目升级

- 先确认活动项目工作区只存在已知用户修改。
- 将 `codex/multi-engine-ocr` 快进合并到本地 `main`，不重置、不覆盖用户修改。
- 使用锁定依赖同步当前项目环境，安装 `semantic` 与 `ocr` 可选依赖。
- 从当前源码构建并验证 Apple Vision helper，要求 arm64、有效本机签名、schema 2、支持 `zh-Hans` 与 `en-US`。
- 将 RapidOCR wheel 自带的三个固定 ONNX 模型复制到 `data/ocr-models/rapidocr`。
- 保持 `BOOK_LIBRARY_ROOT` 和 `BOOK_LIBRARY_OBSIDIAN_VAULT` 不变，使原数据库、原书籍和原笔记继续可用。

### OCR 失败处理

- Apple Vision 页面失败、输出质量不足或边界框异常时，路由器渲染受限尺寸图片并调用 RapidOCR。
- 渲染器在计算和实际 pixmap 两层检查像素上限，避免四舍五入越界。
- 单页仍无法识别时记录为 `skipped`，继续后续页，并在 Obsidian `书库/40-OCR报告` 给出缺失页码和原因。
- 升级过程不调用 `start_ocr` 或 `start_pending_ocr`。只有用户后续明确要求时才重新处理失败书籍。

### 可分发安装包

- 使用相同提交、相同 Vision helper、锁定的 `uv` 和语义检索模型生成确定性 ZIP。
- ZIP 安装器首次运行时自动同步 OCR 依赖并复制 RapidOCR 模型，无需朋友手动安装 OCR 软件。
- 包内包含 `docs/word/安装说明.docx` 与 `docs/word/使用说明.docx`。
- 生成 `SHA256SUMS`，本地验证 ZIP 结构、文件清单、可执行权限和校验值后再上传 GitHub Release。
- GitHub Release 指向包含本次修复的分支或稳定提交，线上附件摘要必须与本地一致。

## 数据与安全边界

- 不删除、不移动、不重新导入现有书籍。
- 不修改现有 OCR 原文证据和 AI 读书笔记。
- 不提交数据库、书籍、Vault 内容、绝对私人路径或登录凭证。
- 保留现有失败状态作为审计记录；重新 OCR 由用户明确授权后触发。

## 验证

1. 先运行与渲染边界、Vision 边界框、RapidOCR 回退、OCR worker、安装器和打包器相关的定向测试。
2. 再运行完整测试套件；macOS Vision 系统测试如遇系统框架偶发错误，单独复跑并如实记录。
3. 验证当前活动配置仍指向原书库根目录和原 Obsidian Vault。
4. 验证活动 Python 环境能够加载 `VisionOcrEngine`、`RapidOcrEngine` 与 `LocalOcrRouter`。
5. 验证 helper capabilities 为 schema 2。
6. 验证 RapidOCR 三个模型均存在且非空。
7. 验证最终 ZIP 和 GitHub Release 的 SHA-256 完全一致。

## 成功标准

- 当前活动项目实际运行双引擎 OCR，而不是仅在另一个工作分支存在新版代码。
- 两类截图错误都进入回退或单页跳过逻辑，不再直接终止整本书。
- 用户现有书库数据和 Obsidian 路径保持不变。
- 朋友只需下载 ZIP 并运行安装脚本，或让 Codex 按说明安装，无需理解 OCR 技术细节。
