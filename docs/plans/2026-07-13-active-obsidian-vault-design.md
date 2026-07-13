# 当前 Obsidian 仓库直连设计

日期：2026-07-13  
状态：已由用户确认方案，等待文档复核

## 背景与根因

用户当前打开的 Obsidian 仓库是：

`/Users/zhaoyunfei/Documents/Obsidian_workspace`

现有 RAG 框架却把 Obsidian 仓库固定为 Codex 项目下的：

`/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo/vault`

因此“书库”虽然已经在项目内部创建，但不会出现在用户当前 Obsidian 的左侧文件列表中。这不是 Obsidian 刷新问题，而是两个独立目录之间的配置错位。

## 目标

1. 当前 Obsidian 左侧直接显示 `书库`。
2. 以后用户只在 Codex 中上传书籍、检索、引用、解释和保存笔记。
3. 原书、解析文本和 AI 读书笔记进入当前 Obsidian 仓库。
4. SQLite 检索数据库和本地嵌入模型继续留在 Codex 项目中。
5. 保持现有引用格式、检索边界和路径安全保护。

## 非目标

- 不要求用户切换或重新打开另一个 Obsidian 仓库。
- 不使用符号链接把两个书库拼接起来。
- 不把模型和 SQLite 数据库放进 Obsidian。
- 不迁移或删除项目内现有的空 `vault/书库` 骨架。
- 不改变 RAG 的分段、向量检索、证据引用和 token 限额策略。

## 选定方案

增加一个独立的 Obsidian 仓库配置：

`BOOK_LIBRARY_OBSIDIAN_VAULT=/Users/zhaoyunfei/Documents/Obsidian_workspace`

保留现有项目根配置：

`BOOK_LIBRARY_ROOT=/Users/zhaoyunfei/Documents/Codex/2026-07-12/wo`

两者职责分离：

```text
当前 Obsidian 仓库
└── 书库
    ├── 00-待导入
    ├── 10-原始书籍
    ├── 20-解析文本
    └── 30-AI读书笔记

Codex 项目
└── data
    ├── library.sqlite3
    └── models
```

如果没有提供新的 Obsidian 配置，程序仍使用原来的项目内 `vault`，从而保持测试和其他部署的向后兼容性。正式 Codex 配置会明确提供当前活动仓库路径。

## 路径与安全边界

程序需要维护两条互相独立的受管根目录：

- Obsidian 根：只允许写入 `书库/00-待导入`、`10-原始书籍`、`20-解析文本` 和 `30-AI读书笔记`。
- 项目根：只允许管理 `data/library.sqlite3`、`data/models` 等项目数据。

所有书库文件写入必须以 Obsidian 根为安全边界；数据库和模型操作仍以项目根为安全边界。现有的反符号链接、目录穿越和竞态保护继续生效。不能简单把所有路径统一切换到 Obsidian，否则会把模型和数据库混入笔记仓库；也不能只改显示路径，否则安全检查会拒绝外部写入。

## 数据流

1. 用户在 Codex 对话中上传一本书。
2. `import_book` 把原文件复制到当前 Obsidian 的 `书库/10-原始书籍`。
3. 解析结果写到 `书库/20-解析文本/<book_id>/正文.md`，段落锚点和 Obsidian wiki-link 规则保持不变。
4. 索引和向量数据写入 Codex 项目的 `data/library.sqlite3`；嵌入模型从项目的 `data/models` 加载。
5. Codex 搜索时先返回少量候选段落，再按需提取原文上下文，继续控制 token 消耗。
6. `save_reading_note` 把带原文引用的笔记写入当前 Obsidian 的 `书库/30-AI读书笔记`。
7. Obsidian 的文件监听器自动显示新目录和文件，无需切换仓库。

## 错误处理

- Obsidian 仓库路径不存在、不可访问或不是安全目录时，启动或操作应给出明确错误。
- 不允许静默回退到旧项目书库，以免形成两个用户难以察觉的“书库”。
- 导入失败时继续遵守现有的原子写入和回滚规则，不能留下半本书或不完整索引。
- 外部 Obsidian 写入失败不能损坏项目数据库的既有数据。

## 测试设计

实现采用测试驱动方式，至少覆盖：

1. 项目根与 Obsidian 根位于两个独立临时目录时，路径计算正确。
2. 原书、解析文本和 AI 笔记只写入外部 Obsidian 根。
3. SQLite 数据库和模型目录只留在项目根。
4. MCP 能从独立环境变量读取当前 Obsidian 路径。
5. 未提供新变量时保持旧默认行为。
6. Obsidian 根、`书库`、解析目录或笔记目录包含符号链接时继续拒绝逃逸写入。
7. 全量自动化测试通过。
8. 在真实当前仓库中启动一次 MCP，创建 `书库` 布局并核对文件树。
9. 最后实际检查 Obsidian 左侧能看到 `书库`。

## 上线与回退

上线时只更新项目代码、测试、Codex MCP 配置和用户指南，然后由 MCP 在当前 Obsidian 仓库中创建一次目录布局。旧项目内 `vault/书库` 保留但不再被正式配置使用，避免进行破坏性删除。

如需回退，只需移除 `BOOK_LIBRARY_OBSIDIAN_VAULT` 配置，程序便恢复到项目内 `vault`；当前 Obsidian 中已经导入的文件不会被自动删除或迁移。
