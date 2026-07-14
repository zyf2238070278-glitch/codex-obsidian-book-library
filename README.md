# Codex Obsidian 本地书库

这是一个给 Codex Desktop 使用的本地书库助手。你可以在 Codex 对话中导入书籍、检索原文、获得通俗解释、比较多本书，并在明确要求后把读书笔记保存到 Obsidian。

当前测试版只面向 Apple Silicon Mac。它支持 PDF、EPUB、Markdown 和 TXT；扫描版 PDF 可在用户明确确认后使用本机 Apple Vision OCR。

## 终端一行安装（推荐）

打开 Mac 的“终端”，完整复制下面这一行并按回车：

```bash
git clone https://github.com/zyf2238070278-glitch/codex-obsidian-book-library.git && cd codex-obsidian-book-library && ./install-from-github.command
```

脚本会自动下载固定版本的全量安装包，核对 SHA-256，并安装到 `~/CodexBookLibrary`。看到“安装完成”后，完全退出并重启 Codex，再在 Codex 中打开并信任 `~/CodexBookLibrary` 整个目录。新建任务并说“检查书库状态”即可。

首次运行需要联网，下载约 292 MB；模型已包含在安装包中。当前命令只支持 Apple Silicon Mac。

## 手动下载 ZIP：三分钟开始

1. 把下载的 ZIP **解压到一个稳定位置**。不要之后随意移动这个目录。请确认是 Apple Silicon Mac、首次安装需要联网，并建议至少预留约 3 GB 可用空间。
2. 双击 `install-macos.command`。如果 macOS 拦截，只使用系统提供的安全入口：在 Finder 中右键或按住 Control 点按该文件，选择“打开”；也可以到“系统设置”→“隐私与安全性”选择“仍要打开”。不要降低整台 Mac 的安全设置。
3. 等待窗口显示“安装完成”。随后完全退出并重启 Codex，仅关闭当前任务或窗口不够。
4. 在 Codex 中打开整个解压目录作为项目，并选择信任项目。只有信任项目后，Codex 才会加载项目里的 `.codex/config.toml`；不要只打开里面的 `Obsidian书库` 子目录。
5. 新建任务并说“检查书库状态”。状态正常后，在 Codex 附加 PDF、EPUB、Markdown 或 TXT，再说“导入这本书”。扫描版 PDF 导入后不会自动 OCR；确认后说“开始 OCR 这本书”。对话和文件附加都在 Codex 完成；Obsidian 只用来浏览书库与笔记。

更完整的步骤见[安装说明](docs/安装说明.md)，日常例句见[使用说明](docs/使用说明.md)。

## 书库放在哪里

默认 Obsidian Vault 是 `<解压目录>/Obsidian书库`。安装器会创建：

- `书库/00-待导入`
- `书库/10-原始书籍`
- `书库/20-解析文本`
- `书库/30-AI读书笔记`

如果要使用已有 Vault，请先在终端进入解压目录，再运行：

```bash
./install-macos.command --vault "<已有 Vault 的绝对路径>"
```

配置中记录的是本机绝对路径。如果移动了解压目录，请在新位置重新运行安装器，然后完全退出并重启 Codex，再重新打开并信任整个项目。

## 重要边界

- 全量安装包内置语义模型，语义检索在本机运行；首次安装 Python 与项目依赖仍需要联网。
- 原书、索引数据库和笔记默认留在本机。你主动附加给 Codex 的文件，以及回答问题时选中的少量短段落，会进入 Codex 对话。
- 问答会使用 Codex 的上下文和 token，因此这不是完全离线、零 token 或零内容传输的方案。
- 当前版本不支持 Intel Mac 或 Windows，也不保证每台 Mac 都不会出现 Gatekeeper 确认。
- OCR 在本机 Apple Vision 上运行，不上传原 PDF，也不需要单独的 OCR token；识别结果会保存页级 checkpoint，可能有错字，引用前应按 PDF 物理页核验。

更多细节见[隐私与数据存放](docs/隐私与数据存放.md)和[常见问题](docs/常见问题.md)。项目源码采用 [MIT License](LICENSE)，随包组件见[第三方说明](THIRD_PARTY_NOTICES.md)。
