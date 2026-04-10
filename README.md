# Zhihu-CSM-Reply-Robot

> 知乎 CSM（客户成功管理）/LabVIEW 专栏自动回复机器人 —— 基于 RAG + DeepSeek LLM，运行于 GitHub Actions

---

## 功能概览

- 📥 每15分钟定时拉取知乎文章/问题/专栏下的新评论（**不使用 LLM**，仅规则过滤，节省 token）
- 🔍 RAG 检索 CSM Wiki 知识库，结合上下文生成专业回复
- 🤖 调用 DeepSeek（或其他 OpenAI 兼容模型）生成回复，回复统一加 `[rob]:` 前缀标识自动回复
- 🧠 **AI 自动风险判断**：关于 CSM/LabVIEW 的明确回复直接自动发布；LLM 判定需人工介入的高危回复才写入 `pending/`
- 👤 **白名单用户过滤**：维护者等白名单用户的评论仅记录，不触发 AI 处理，节省 token
- 📚 **回复自学习**：所有回复内容（bot 回复 + 人工回复）自动加入 RAG 索引，持续提升回复质量
- 📝 **文章摘要记录**：线程记录中使用 LLM 生成的简短摘要（而非全文），便于 AI 理解上下文
- 🚨 异常自动告警：Cookie 失效、429 限流、预算超限 → 创建 GitHub Issue
- 💰 每日 LLM 费用追踪与预算限制
- 📊 追问上下文管理（多轮对话线程）

---

## 快速开始

### 1. Fork 仓库

点击右上角 **Fork** 按钮，将仓库 Fork 到自己的 GitHub 账号下。

### 2. 配置监控目标

编辑 `config/articles.yaml`，支持监控**单篇文章、知乎问题、整个专栏，或某人发布的所有回答**：

```yaml
articles:
  # 单篇专栏文章
  - id: "98765432"
    title: "CSM 最佳实践系列（一）"
    url: "https://zhuanlan.zhihu.com/p/98765432"
    type: "article"

  # 知乎问题（监控该问题下所有回答的评论）
  - id: "123456789"
    title: "如何做好客户成功？"
    url: "https://www.zhihu.com/question/123456789"
    type: "question"

  # 整个专栏（自动监控专栏内全部文章）
  - id: "csm-practice"
    title: "CSM 实践专栏"
    url: "https://www.zhihu.com/column/csm-practice"
    type: "column"

  # 某人发布的全部回答
  - id: "nevstop"
    title: "nevstop 的全部回答"
    url: "https://www.zhihu.com/people/nevstop/answers"
    type: "user_answers"
```

### 3. 准备 CSM Wiki 知识库

将 Markdown 格式的 CSM/LabVIEW 知识文档放入 `csm-wiki/` 目录：

```
csm-wiki/
├── 01-客户成功概述.md
├── 02-NPS与CSAT指标.md
├── 03-客户健康度模型.md
└── ...
```

### 4. 配置 GitHub Secrets

在仓库 **Settings → Secrets and variables → Actions** 中添加以下 Secrets：

| Secret 名称 | 必填 | 说明 |
|---|---|---|
| `ZHIHU_COOKIE` | ✅ | 知乎完整 Cookie 字符串（含 `z_c0` 和 `_xsrf`） |
| `LLM_API_KEY` | ✅ | DeepSeek 或 OpenAI 兼容服务的 API Key |
| `LLM_BASE_URL` | ❌ | LLM API 端点，默认 `https://api.deepseek.com` |
| `LLM_MODEL` | ❌ | 模型名称，默认 `deepseek-chat` |
| `GITHUB_TOKEN` | 自动 | GitHub Actions 自动注入，用于告警创建 Issue |

> **说明**：以上所有敏感信息（Cookie、LLM Key 等）均通过 GitHub Secrets 传入，不在代码或配置文件中明文保存。

#### 获取知乎 Cookie

1. 浏览器登录知乎
2. 打开开发者工具（F12）→ Network 面板 → 刷新页面
3. 点击任意知乎请求 → Headers → 复制 `Cookie` 字段的完整值

### 5. 调整运行参数（可选）

编辑 `config/settings.yaml`：

```yaml
bot:
  max_new_comments_per_run: 20     # 每次最多处理条数
  max_new_comments_per_day: 100    # 每日上限
  llm_budget_usd_per_day: 0.50    # 每日 LLM 费用预算（超出后停止并告警）
  reply_prefix: "[rob]"            # 回复前缀，让用户知道这是自动回复
  whitelist_users:                 # 白名单用户（维护者等），仅记录不做 AI 处理
    - "your-zhihu-username"

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
| `bot.yml` | **每15分钟** + 手动触发 | 拉取评论 → AI生成回复 → 自动发布或写入 pending/ |
| `sync-wiki.yml` | 每周日 + 手动触发 | 增量同步 CSM Wiki 向量库 |

### bot.yml 所需 Secrets

```yaml
env:
  ZHIHU_COOKIE:   ${{ secrets.ZHIHU_COOKIE }}    # 知乎 Cookie（必填）
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
Zhihu-CSM-Reply-Robot/
├── .github/workflows/
│   ├── bot.yml              # 主 Workflow（每15分钟定时回复）
│   └── sync-wiki.yml        # Wiki 同步 Workflow
├── config/
│   ├── settings.yaml        # 全局运行参数
│   └── articles.yaml        # 监控目标列表（文章/问题/专栏/用户）
├── csm-wiki/                # CSM/LabVIEW 知识库 Markdown 文档（自行添加）
├── data/
│   ├── seen_ids.json        # 已处理评论 ID 记录
│   ├── vector_store/        # ChromaDB 向量库（自动生成）
│   └── reply_index/         # 历史回复向量索引
├── pending/                 # AI 判定需人工审核的高危回复
├── archive/                 # 已发布/归档回复
├── scripts/
│   ├── run_bot.py           # 主入口
│   ├── wiki_sync.py         # Wiki 同步 CLI
│   ├── zhihu_client.py      # 知乎 API 封装
│   ├── rag_retriever.py     # RAG 检索模块
│   ├── llm_client.py        # LLM 调用模块
│   ├── thread_manager.py    # 多轮对话管理
│   ├── comment_filter.py    # 评论前置过滤
│   ├── alerting.py          # GitHub Issue 告警
│   ├── cost_tracker.py      # 费用追踪
│   └── archiver.py          # 归档管理
└── tests/                   # 单元测试（174 个）
```

---

## 回复发布流程

机器人采用 **AI 自动风险判断**，而非全量人工审核：

```
新评论
  │
  ▼
白名单检查 ──→ 白名单用户 ──→ 仅记录到线程 + RAG（不做 AI 处理）
  │
  ▼
规则过滤（垃圾/广告/重复）
  │  不使用 LLM，节省 token
  ▼
RAG 检索知识库 + LLM 生成回复
  │  回复统一加 [rob]: 前缀
  │  回复内容自动加入 RAG 学习
  ▼
LLM 风险评估
  ├─ 明确的 CSM/LabVIEW 回复 ──→ 直接自动发布 ✅
  └─ 需人工判断（敏感/超出知识库）──→ 写入 pending/ 等待审核 📝
```

**审核并发布 pending/ 中的回复**：

1. 打开 `pending/` 目录中对应的 `.md` 文件
2. 修改内容（可选）
3. 将文件中 `status: pending` 改为 `status: approved`
4. 提交到仓库，下次 Workflow 运行时自动发布

---

## 告警机制

以下异常会自动在 GitHub 仓库创建 Issue（标签 `bot-alert`）：

| 告警类型 | 触发条件 |
|---|---|
| Cookie 失效 | HTTP 401 / 403 |
| 持续限流 | HTTP 429 重试耗尽 |
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
python -m pytest tests/test_zhihu_client.py -v
python -m pytest tests/test_llm_client.py -v
```

