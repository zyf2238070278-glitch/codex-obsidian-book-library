# Codex Obsidian 本地书库

这是一个给 Codex Desktop 使用的本地书库助手。你可以在 Codex 对话中导入书籍、检索原文、获得通俗解释、比较多本书，并在明确要求后把读书笔记保存到 Obsidian。

当前版本支持 macOS 16 或更高版本的 Apple Silicon（M 系列）Mac，不支持 Intel Mac 或 Windows。支持 PDF、EPUB、Markdown 和 TXT；扫描版 PDF 可在用户明确授权后按 Apple Vision、RapidOCR、Light OCR 的顺序在本机识别。

## Git 安装（终端一行，推荐）

1. 打开终端，完整复制下面这一行并按回车；它会克隆仓库并自动运行安装器：

   ```bash
   git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git && cd codex-obsidian-book-library && ./install-from-github.command
   ```

2. 看到“安装完成”后，在 Codex 中打开并信任整个 `codex-obsidian-book-library` 项目。如果安装曾中断，也可以新建任务说：“请安装并检查这个书库”。
3. 完整退出并重启 Codex，再打开同一项目。
4. 重启后新建任务并说：“检查书库状态”。状态正常后，可以附加书籍并说“导入这本书”。

这条流程只支持 macOS 16 或更高版本的 Apple Silicon（M 系列）Mac。首次安装需要联网：安装器会下载约 500 MB 的语义模型及 Python 包，并下载经过 SHA-256 校验的固定 Node.js 与 Light OCR 运行包；实际流量会随包缓存状态略有变化。项目自带固定版本的 uv、Apple Vision 工具，并在项目内创建项目本地 Python、安装锁定版本的依赖以及准备 RapidOCR 模型；本机无需预装 Homebrew、Python、Xcode 或 uv，也无需预装 Node.js。安装完成后，语义检索、Apple Vision、RapidOCR 和 Light OCR 都在本机运行。

一行命令中的 `install-from-github.command` 只会转交给项目根目录的 `install-macos.command`。如果需要手动重试，也可以在项目根目录运行任一入口；不要自行拼接 Python、pip、uv、npm 或模型下载命令。

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
- `书库/50-书目卡片`

`书库/书库总览.base` 可以直接按主分类查看全部书籍、待 OCR 书籍和有 OCR 警告的书籍。书目卡片里的 `primary_category` 与 `custom_categories` 可以在 Obsidian 中自行修改；后续同步会保留自定义分类。

运行环境、数据库和模型位于项目内的 `.venv` 与 `data/`，本机配置写入 `.codex/config.toml`。如果要使用已有 Vault，可在项目目录运行：

```bash
./install-macos.command --vault "<已有 Vault 的绝对路径>"
```

## 移动、修复与备份

配置中记录本机绝对路径。移动项目目录后，在 Codex 中打开并信任新位置，再说“请安装并检查这个书库”，安装成功后完整退出并重启 Codex。重新运行安装器会修复环境和刷新配置，不会删除已有书籍或笔记。

默认 Vault 位于项目目录内；删除整个项目目录可能同时删除原书、解析文本和笔记，删除或迁移前请先备份。详情见[安装说明](docs/安装说明.md)、[隐私与数据存放](docs/隐私与数据存放.md)和[常见问题](docs/常见问题.md)。

OCR 完全在本机运行：默认使用 Apple Vision，失败或质量不足时依次切换 RapidOCR、Light OCR；不上传原 PDF，也不需要单独的 OCR token。识别结果会保存页级 checkpoint，可能有错字，引用前应按 PDF 物理页核验。

## 隐私边界

原书、索引数据库、模型和笔记默认留在本机。你主动附加给 Codex 的文件，以及回答问题时选中的少量短段落，会进入 Codex 对话。问答会使用 Codex 上下文和 token，因此这不是完全离线、零 token 或零内容传输的方案。项目源码采用 [MIT License](LICENSE)，随包组件见[第三方说明](THIRD_PARTY_NOTICES.md)。
