# CSM-Reply-Robot

> CSM（Communicable State Machine，通信状态机）/LabVIEW 问答自动回复机器人 —— 基于 RAG + DeepSeek LLM，运行于 GitHub Actions

---

## 功能概览

- 📥 读取 `data/inbox/` 目录中的待处理消息（平台无关，由外部工具/脚本写入）
- 🔍 RAG 检索 CSM Wiki 知识库，结合上下文生成专业回复
- 🤖 调用 DeepSeek（或其他 OpenAI 兼容模型）生成回复，回复统一加 `[rob]:` 前缀标识自动回复
- 📝 所有 AI 回复写入 `data/pending/` 供人工审核后发布
- 👤 **白名单用户过滤**：维护者等白名单用户的消息仅记录，不触发 AI 处理，节省 token
- 📚 **回复自学习**：所有回复内容（bot 回复 + 人工回复）自动加入 RAG 索引，持续提升回复质量
- 🚨 异常自动告警：预算超限、连续失败 → 创建 GitHub Issue
- 💰 每日 LLM 费用追踪与预算限制
- 📊 追问上下文管理（多轮对话线程）

---

## 消息输入格式

机器人从 `data/inbox/` 目录读取 JSON 文件，每个文件代表一条待处理消息：

```json
{
  "id": "msg-001",
  "content": "CSM 框架如何处理多状态并发？",
  "author": "username",
  "created_time": 1700000000,
  "parent_id": null,
  "is_author_reply": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 消息唯一 ID |
| `content` | string | 消息正文 |
| `author` | string | 发送者用户名 |
| `created_time` | int | Unix 时间戳（秒） |
| `parent_id` | string \| null | 父消息 ID（追问时填写） |
| `is_author_reply` | bool | 是否为专家/维护者回复（仅索引，不生成 AI 回复） |

---

## 配置

### 1. 准备 CSM Wiki 知识库

`csm-wiki/` 目录用于存放**本地补充文档**（可选）。主要知识库来源是 [CSM Wiki](https://nevstop-lab.github.io/CSM-Wiki/)，由 `sync-wiki.yml` 工作流自动从 [NEVSTOP-LAB/CSM-Wiki](https://github.com/NEVSTOP-LAB/CSM-Wiki) 拉取并索引。

本地 `csm-wiki/` 目录可用于放置私有补充文档：

```
csm-wiki/
├── 私有补充文档.md
└── ...
```

### 2. 配置 GitHub Secrets

在仓库 **Settings → Secrets and variables → Actions** 中添加以下 Secrets：

| Secret 名称 | 必填 | 说明 |
|---|---|---|
| `LLM_API_KEY` | ✅ | DeepSeek 或 OpenAI 兼容服务的 API Key |
| `LLM_BASE_URL` | ❌ | LLM API 端点，默认 `https://api.deepseek.com` |
| `LLM_MODEL` | ❌ | 模型名称，默认 `deepseek-chat` |
| `GITHUB_TOKEN` | 自动 | GitHub Actions 自动注入，用于告警创建 Issue |

> **说明**：所有敏感信息均通过 GitHub Secrets 传入，不在代码或配置文件中明文保存。

### 3. 调整运行参数（可选）

编辑 `config/settings.yaml`：

```yaml
bot:
  max_new_comments_per_run: 20     # 每次最多处理条数
  max_new_comments_per_day: 100    # 每日上限
  llm_budget_usd_per_day: 0.50    # 每日 LLM 费用预算（超出后停止并告警）
  reply_prefix: "[rob]"            # 回复前缀，让用户知道这是自动回复
  whitelist_users:                 # 白名单用户（维护者等），仅记录不做 AI 处理
    - "your-username"

filter:
  spam_keywords:                   # 广告关键词（命中则跳过）
    - "加微信"
    - "私信"
```

---

## GitHub Actions 配置

仓库内置两个 Workflow，Fork 后即可直接使用：

| Workflow | 触发方式 | 功能 |
|---|---|---|
| `bot.yml` | **每15分钟** + 手动触发 | 读取 inbox → AI 生成回复 → 写入 pending/ |
| `sync-wiki.yml` | 每周日 + 手动触发 | 从 [NEVSTOP-LAB/CSM-Wiki](https://github.com/NEVSTOP-LAB/CSM-Wiki) 拉取最新文档并增量同步向量库 |

### bot.yml 所需 Secrets

```yaml
env:
  LLM_API_KEY:    ${{ secrets.LLM_API_KEY }}      # LLM API Key（必填）
  LLM_BASE_URL:   ${{ secrets.LLM_BASE_URL }}     # LLM 端点（可选）
  LLM_MODEL:      ${{ secrets.LLM_MODEL }}        # 模型名称（可选）
  GITHUB_TOKEN:   ${{ secrets.GITHUB_TOKEN }}     # 自动注入，用于告警
```

### 启用 / 停用 Workflow

- 启用：仓库页面 → **Actions** → 选择 Workflow → **Enable workflow**
- 手动触发：Actions → 选择 Workflow → **Run workflow**
- 停用：Actions → 选择 Workflow → **Disable workflow**

---

## 目录结构

```
CSM-Reply-Robot/
├── .github/workflows/
│   ├── bot.yml              # 主 Workflow（每15分钟定时回复）
│   └── sync-wiki.yml        # Wiki 同步 Workflow
├── config/
│   └── settings.yaml        # 全局运行参数
├── csm-wiki/                # 本地补充文档（可选，主库由 sync-wiki.yml 自动从远程拉取）
├── data/
│   ├── inbox/               # 待处理消息（JSON，由外部工具写入）
│   ├── pending/             # AI 生成的回复（等待人工审核）
│   ├── done/                # 已审核通过的回复
│   ├── vector_store/        # ChromaDB 向量库（自动生成）
│   └── reply_index/         # 历史回复向量索引
├── archive/                 # 对话线程归档（自动生成）
├── scripts/
│   ├── run_bot.py           # 主入口
│   ├── wiki_sync.py         # Wiki 同步 CLI
│   ├── rag_retriever.py     # RAG 检索模块
│   ├── llm_client.py        # LLM 调用模块
│   ├── thread_manager.py    # 多轮对话管理
│   ├── comment_filter.py    # 消息前置过滤
│   ├── alerting.py          # GitHub Issue 告警
│   ├── cost_tracker.py      # 费用追踪
│   └── archiver.py          # 归档管理
└── tests/                   # 单元测试
```

---

## 回复处理流程

```
data/inbox/ 中的消息
  │
  ▼
白名单检查 ──→ 白名单用户 ──→ 仅记录到线程 + RAG（不做 AI 处理）
  │
  ▼
规则过滤（垃圾/广告/重复）
  │
  ▼
RAG 检索知识库 + LLM 生成回复
  │  回复统一加 [rob]: 前缀
  │  回复内容自动加入 RAG 学习
  ▼
写入 data/pending/ 等待人工审核 📝
```

**审核 pending/ 中的回复**：

1. 打开 `data/pending/` 目录中对应的 `.md` 文件，确认回复内容
2. 调用 `archiver.approve_pending(filepath)` 将文件移至 `data/done/`（或手动移动）

---

## 告警机制

以下异常会自动在 GitHub 仓库创建 Issue（标签 `bot-alert`）：

| 告警类型 | 触发条件 |
|---|---|
| 连续失败 | 连续失败 ≥ 3 次（可配置） |
| 预算超限 | 当日 LLM 费用 > 预算上限 |

---

## 开发与测试

```bash
# 安装依赖
pip install -r requirements.txt

# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_run_bot.py -v
python -m pytest tests/test_llm_client.py -v
```

