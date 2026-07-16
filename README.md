# Codex Obsidian 本地书库

这是一个给 Codex Desktop 使用的本地书库助手。你可以在 Codex 对话中导入书籍、检索原文、获得通俗解释、比较多本书，并在明确要求后把读书笔记保存到 Obsidian。

当前版本支持 macOS 16 或更高版本的 Apple Silicon（M 系列）Mac，不支持 Intel Mac 或 Windows。支持 PDF、EPUB、Markdown 和 TXT；扫描版 PDF 可在用户明确授权后使用本机 Apple Vision，并在需要时切换到本地 RapidOCR。

## Git 安装（推荐）

1. 在终端运行精确的克隆命令，然后进入项目目录：

   ```bash
   git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git
   cd codex-obsidian-book-library
   ```

2. 在 Codex 中打开并信任整个 `codex-obsidian-book-library` 项目，新建任务并说：“请安装并检查这个书库”。
3. 安装成功后，完整退出并重启 Codex，再打开同一项目。
4. 重启后新建任务并说：“检查书库状态”。状态正常后，可以附加书籍并说“导入这本书”。

这条流程只支持 macOS 16 或更高版本的 Apple Silicon（M 系列）Mac。首次安装需要联网：安装器会下载约 500 MB 的语义模型及 Python 包，实际流量会随包缓存状态略有变化。项目自带固定版本的 uv、Apple Vision 工具，并在项目内创建项目本地 Python、安装锁定版本的依赖以及准备 RapidOCR 模型；本机无需预装 Homebrew、Python、Xcode 或 uv。安装完成后，语义检索、Apple Vision 和 RapidOCR 都在本机运行。

Codex 只运行项目根目录的 `install-macos.command`。如果你需要手动重试，也可以在项目根目录运行它；不要自行拼接 Python、pip、uv 或模型下载命令。

## 书库怎么用

在 Codex 中附加 PDF、EPUB、Markdown 或 TXT 后说“导入这本书”。需要原文时明确说“引用原文”，Codex 会先检索，再展开少量相关段落并标注《书名》和 PDF 页、EPUB 章节或 `passage_id`。需要通俗解释或跨书比较时，也会先核对原书证据。

扫描版 PDF 导入后只标记为待 OCR，不会自动开始。只有你明确说“开始 OCR 这本书”或“处理所有待 OCR 书籍”时才会启动；OCR 结果可能有误，引用前仍需核对 PDF 物理页。只有你明确要求“保存”时，Codex 才会把已核验内容保存为 AI 读书笔记。Obsidian 只用来浏览原书、解析文本和笔记。

更完整的例句见[使用说明](docs/使用说明.md)和[使用指南](docs/USER_GUIDE.md)。

## 数据放在哪里

默认 Obsidian Vault 是 `<项目目录>/Obsidian书库`。安装器会创建：

- `书库/00-待导入`
- `书库/10-原始书籍`
- `书库/20-解析文本`
- `书库/30-AI读书笔记`
- `书库/40-OCR报告`

运行环境、数据库和模型位于项目内的 `.venv` 与 `data/`，本机配置写入 `.codex/config.toml`。如果要使用已有 Vault，可在项目目录运行：

```bash
./install-macos.command --vault "<已有 Vault 的绝对路径>"
```

## 移动、修复与备份

配置中记录本机绝对路径。移动项目目录后，在 Codex 中打开并信任新位置，再说“请安装并检查这个书库”，安装成功后完整退出并重启 Codex。重新运行安装器会修复环境和刷新配置，不会删除已有书籍或笔记。

默认 Vault 位于项目目录内；删除整个项目目录可能同时删除原书、解析文本和笔记，删除或迁移前请先备份。详情见[安装说明](docs/安装说明.md)、[隐私与数据存放](docs/隐私与数据存放.md)和[常见问题](docs/常见问题.md)。

## 隐私边界

原书、索引数据库、模型和笔记默认留在本机。你主动附加给 Codex 的文件，以及回答问题时选中的少量短段落，会进入 Codex 对话。问答会使用 Codex 上下文和 token，因此这不是完全离线、零 token 或零内容传输的方案。项目源码采用 [MIT License](LICENSE)，随包组件见[第三方说明](THIRD_PARTY_NOTICES.md)。
