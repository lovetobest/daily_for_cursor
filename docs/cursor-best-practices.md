# Cursor 最佳实践

面向用 Cursor 写代码、改仓库、跑 Cloud Agent 的日常工作。原则来自 [官方 Agent 指南](https://cursor.com/blog/agent-best-practices)、[Rules](https://cursor.com/docs/rules.md)、[Skills](https://cursor.com/docs/skills.md) 和 [Cloud Agent](https://cursor.com/docs/cloud-agent/best-practices.md)。本仓库已经按这些原则配好了 `AGENTS.md`、`.cursor/rules/` 和 `.cursor/skills/`。

把 Agent 当成**聪明、但缺少项目上下文的同事**：它能搜代码、改文件、跑命令，但不会记住上次对话，也不会自动知道你们团队的约定。你的工作是给它**可验证的目标**和**刚好够用的上下文**。

## 1. 先规划，再写代码

复杂改动先开 **Plan Mode**（Agent 输入框里 `Shift+Tab`）。Agent 会先读仓库、问澄清问题、给出带文件路径的计划，等你批准再动手。

适合用计划的场景：

- 跨多个文件的重构
- 行为不明确、有几种实现路径
- 你自己也还没想清楚接口和边界

不适合：改一个 typo、加一个测试、你做过很多次的机械活。

计划会变成可编辑的 Markdown。不对就改计划，而不是在错误实现上追问。改歪了就回滚，收紧计划再跑一遍，通常比在半成品上打补丁更快。需要跨会话接着做时，把计划存到 `.cursor/plans/`。

## 2. 管好上下文

上下文窗口是固定的。文件、工具结果、规则、MCP、历史消息都占额度。窗口快满时 Cursor 会压缩旧对话，Agent 开始跑偏、重复犯错时，多半该新开一轮。

**新开对话**当：换了一个功能、Agent 开始循环犯错、一个逻辑单元已经做完。

**接着聊**当：还在同一功能上迭代、需要上一轮的结论、正在调试它刚写的代码。

给上下文的方式：

| 做法 | 何时用 |
|------|--------|
| 不 @，让 Agent 自己搜 | 不确定相关文件在哪 |
| `@文件` / `@目录` | 你已经知道关键路径 |
| `@Branch` / `@Commit` | 审查当前改动 |
| `@Chats` | 新对话需要上一轮结论，不要整段粘贴 |
| `@Terminals` | 失败日志已经在终端里 |
| 截图 / 设计稿 | UI、报错画面、视觉回归 |

少即是多：把无关文件塞进上下文，等于告诉模型「这些也很重要」。

输入框旁的圆环能看到窗口占用。点开可按 System prompt / Tools / Rules / Skills / MCP / 对话 查看各占多少。规则和 MCP 堆太多，有效对话空间会被挤掉。

## 3. 把项目知识写进仓库

不要把约定只放在某次聊天里。下次对话、下一个同事、Cloud Agent 都读不到。

### 选哪一种

| 机制 | 放什么 | 何时加载 |
|------|--------|----------|
| `AGENTS.md` | 项目是什么、怎么跑、目录约定 | 该目录树内自动生效 |
| `.cursor/rules/*.mdc` | 稳定、短、可执行的约束 | always / glob / 按描述 / 手动 `@` |
| `.cursor/skills/*/SKILL.md` | 可重复的流程（发布、加 sample、跑测试） | Agent 判断相关，或你输入 `/skill-name` |
| 用户规则 | 你个人的语气、偏好 | 所有仓库的 Agent Chat |
| 团队规则 | 公司级规范 | Dashboard，可强制执行 |

经验规则：

- **Always-on 的短事实** → `AGENTS.md` 或 `alwaysApply` 规则
- **只对某类文件成立** → 带 `globs` 的规则
- **步骤超过一段、偶尔才用** → Skill（正文按需加载，不占每次对话）
- **Agent 反复犯的同一个错** → 这时才加规则，不要预先堆百科全书

官方建议：单条规则不超过 500 行；大规则拆开；引用文件而不是把文件内容复制进规则；能用 linter 的不要写成文风指南。

不要再用根目录 `.cursorrules`。旧的 slash command（`.cursor/commands/`）可以 `/migrate-to-skills` 迁到 Skills。

### 规则怎么写

`.cursor/rules/` 里必须是 `.mdc`（带 YAML frontmatter）。纯 `.md` 会被忽略。

```markdown
---
description: Python style for this repo's samples
globs: "**/*.py"
alwaysApply: false
---

Match `beamsearch/search.py`. Target Python 3.9+, stdlib only.
```

四种触发：

| `alwaysApply` | `description` | `globs` | 行为 |
|---------------|---------------|---------|------|
| `true` | 任意 | 任意 | 每次对话都带上 |
| `false` | — | 有 | 匹配文件在上下文时自动带上 |
| `false` | 有 | 无 | Agent 根据描述决定 |
| `false` | 无 | 无 | 只有你 `@规则名` 时才带上 |

好规则像内部文档：具体、可执行、指向规范文件。坏规则像愿望清单：「写优雅的代码」「注意性能」。

优先级冲突时：**团队规则 → 项目规则 → 用户规则**。能合并的会合在一起，冲突以更靠前的为准。

### Skill 怎么写

每个 Skill 是一个目录，里面有 `SKILL.md`。`name` 必须和目录名一致。

```markdown
---
name: run-tests
description: Run this repo's Python unit tests and report failures. Use when verifying a change or after editing Python files.
---
```

`description` 决定 Agent 会不会自动用它。写清**做什么**和**何时用**。长参考放到 `references/`，脚本放到 `scripts/`，让 Agent 按需读取。

本仓库的示例：

- `.cursor/skills/run-tests/` — 跑 `unittest`
- `.cursor/skills/add-sample/` — 按 `beamsearch/` 的布局加新 demo

Chat 里输入 `/` 可手动调用。设 `disable-model-invocation: true` 则只有手动 `/name` 才会加载。

## 4. 把 Prompt 写具体

模型对具体指令的成功率明显高于空泛指令。

弱：`给 auth.ts 加测试`

强：`按 tests/ 里现有 unittest 风格，为 beamsearch/search.py 的 length_normalize=True 补一个失败用例，先让测试红，再写实现，不要改已有断言。`

可复用的结构：

1. **目标**：做完后什么算成功
2. **范围**：改哪些路径，哪些不要动
3. **约束**：语言版本、依赖、测试命令
4. **验收**：跑哪条命令、期望看到什么

TDD 对 Agent 特别有效，因为它有红/绿信号可以自己迭代：

1. 让它只写测试，并确认失败
2. 你满意后提交测试
3. 再让它写实现，禁止改测试，直到全绿
4. 提交实现

## 5. 给可验证的完成条件

Agent 不能修它看不见的问题。仓库里有测试、类型检查、linter，它才能闭环。

对本仓库：

```bash
python3 -m unittest discover -s tests -v
```

对更大的项目，把「改完后跑什么」写进 `AGENTS.md`，而不是每次聊天重复。Cloud Agent 尤其依赖这一点：环境里跑不起来的测试，云上的 Agent 也跑不起来。

## 6. 审查 AI 写的代码

生成越快，审查越重要。看起来正确、实际错一点的 diff 很常见。

- 生成过程中看 diff，跑偏就 Stop，改口令再继续
- 结束后用 Review → Find Issues，或在 Source Control 里对主分支做 Agent Review
- PR 上开 Bugbot
- 大改动让 Agent 画一张 Mermaid 数据流图，审查架构而不是逐行死磕

不要把「测试绿了」当成「设计对了」。测试只能证明你写进断言的行为。

## 7. 并行和 Cloud Agent

难问题可以让多个模型同时做，再挑最好的结果。本地可用 git worktree 隔离；Cloud Agent 适合丢进待办的活：修旁边冒出来的 bug、给现有代码补测试、更新文档。你关电脑它也能跑，最后以 PR 回来。

Cloud Agent 想靠谱，顺序是：

1. **先配环境**（依赖能装上、测试能跑）
2. **再配网络和 Secrets**（egress allowlist、密钥、OIDC）
3. **再用 `AGENTS.md` / Skills / MCP** 补项目怎么测、怎么调试

把 Agent 需要的系统和人用的对齐：数据库、日志、issue tracker。工具本身也要适合模型用：参数少、输出干净。人类会忽略的嘈杂构建日志，模型可能会被带跑。

本地 IDE 用的指示 Cloud Agent 也会读。只想给云端看的内容放 `.cursor/CLOUD.md`。

## 8. Debug Mode

普通 Agent 猜不准的 bug（能复现但找不到原因、竞态、性能、回归）用 Debug Mode：先提出假设，再打日志，让你复现，用运行时数据定位，再做小补丁。复现步骤写得越具体，插桩越有用。

## 9. 本仓库怎么落地的

```text
.
├── AGENTS.md                          # 项目事实：命令、约定、工作流
├── .cursor/rules/
│   ├── project.mdc                    # Always Apply：范围和验收
│   ├── python.mdc                     # glob **/*.py
│   └── tests.mdc                      # glob tests/**/*.py
├── .cursor/skills/
│   ├── run-tests/SKILL.md             # /run-tests
│   └── add-sample/SKILL.md            # /add-sample
└── docs/cursor-best-practices.md      # 这份指南
```

加规则的节奏：先跑起来，发现 Agent 反复踩同一个坑，再写一条短规则。不要在第一天把整个风格指南倒进去。

## 官方文档

- [Best practices for coding with agents](https://cursor.com/blog/agent-best-practices)
- [Rules](https://cursor.com/docs/rules.md)
- [Agent Skills](https://cursor.com/docs/skills.md)
- [Prompting](https://cursor.com/docs/agent/prompting.md)
- [Cloud Agent best practices](https://cursor.com/docs/cloud-agent/best-practices.md)
- [Subagents](https://cursor.com/docs/subagents.md)
