# 代码 Review 修改计划（2026-04-10 轮）

> **来源**：汇总自 `docs/Review-Code/` 下本轮三份评审报告
> - `claude-2026-04-10.md`（以下简称 **claude**）
> - `claude-agent-2026-04-10.md`（以下简称 **claude-agent**）
> - `codex - 2026-04-10.md`（以下简称 **codex**）
>
> 每条修改项均标注提出者与原始建议出处，便于追溯。

---

## 一、问题汇总与修改优先级

### 🔴 高优先级（影响功能目标）

---

#### FIX-01：question 类型无法正常读取/发布评论

| 字段 | 内容 |
|------|------|
| **提出者** | codex |
| **原文位置** | `codex - 2026-04-10.md` § 主要发现 1 |
| **涉及代码** | `scripts/zhihu_client.py:148-168`，`scripts/run_bot.py:504-516` / `639-683` |
| **原始描述** | `object_type=question` 时使用 `/answers/{id}/comments` 但传入的是 question id 而非 answer id；`_expand_articles()` 未将 question 展开为具体回答；自动发布将 question 映射为 `object_type="answer"` 但仍传入 question id，导致既无法抓取评论也无法发布回复 |
| **修改方向** | 在 question 场景下先通过 API 获取该问题的回答列表（包括自己的回答），取回 answer id 后再用正确的 answer id 调用读/写接口 |

---

#### FIX-02：RAG 检索使用文章标题而非评论内容，所有评论复用同一份 context

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.6 问题 10 |
| **涉及代码** | `scripts/run_bot.py:315-324` |
| **原始描述** | `retrieve(query=article.get("title", ""))` 用文章标题作为 query，同一文章所有评论复用同一份 context_chunks，忽略每条评论的具体内容差异；计划文档流程图中"RAG 检索"节点位于处理单条评论的路径上，说明意图是按评论内容检索 |
| **修改方向** | 将 RAG 检索移入 `_process_single_comment()`，用当前评论的内容作为 `query` 进行检索，使每条评论获得最相关的 Wiki 片段 |

---

#### FIX-03：Workflow cron 每 15 分钟触发，与计划 6 小时不符

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 3.1 问题 18 |
| **涉及代码** | `.github/workflows/bot.yml`（cron 表达式） |
| **原始描述** | `*/15 * * * *` 每 15 分钟触发，与 `plan/README.md` 和 AI-004 验收条件中 `cron '0 2,8,14,20 * * *'`（每 6 小时）不符；每天消耗约 192 分钟 Actions 额度，可能加速 Cookie 失效 |
| **修改方向** | 将 `bot.yml` cron 表达式改为 `0 2,8,14,20 * * *`，并更新注释；同步更新 `plan/README.md` AI-004 验收条件以保持一致 |

---

#### FIX-04：去重窗口（dedup_window_minutes）无持久化，跨运行实际无效

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.5 问题 9 |
| **涉及代码** | `scripts/comment_filter.py`（`_recent_comments` 字典） |
| **原始描述** | `_recent_comments` 是内存字典，每次 `BotRunner.init_modules()` 都会创建新实例并清空，`dedup_window_minutes: 60` 的配置仅在单次运行内有效；而 bot 每 15 分钟运行一次，实际去重效果接近零 |
| **修改方向** | 将 `_recent_comments` 持久化到 `data/dedup_cache.json`，类似 `seen_ids.json` 的读写方式；初始化时加载，过滤后写入，并在加载时清除过期条目 |

---

### 🟡 中优先级（影响准确性或可维护性）

---

#### FIX-05：`summarize_article` 将 title 作为 content 传入，生成无实质内容的摘要

| 字段 | 内容 |
|------|------|
| **提出者** | claude（问题 11）；codex（主要发现 2） |
| **原文位置** | `claude-2026-04-10.md` § 2.6 问题 11；`codex - 2026-04-10.md` § 主要发现 2 |
| **涉及代码** | `scripts/run_bot.py:305-308` |
| **原始描述** | `content=article.get("title", "")` 将 title 同时作为正文传入，生成的"摘要"只是标题的复述，无法为线程元数据或 RAG 提供背景；当前架构中 articles.yaml 不含正文字段，需额外 API 调用 |
| **修改方向** | 方案 A：调用知乎 API 拉取文章正文后传入（需增加 `get_article_content()` 方法）。方案 B（简洁）：如无法获取正文，则直接省略 `summarize_article` 调用，避免浪费 token 生成无意义摘要，改为直接使用 title 作为 `article_summary` 存入元数据 |

---

#### FIX-06：Bot 回复在风险评估之前已被索引到 RAG，低质量/被拒回复会污染检索

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.6 问题 12 |
| **涉及代码** | `scripts/run_bot.py:485-495`（索引）vs `scripts/run_bot.py:497`（风险评估） |
| **原始描述** | `index_human_reply()` 在 `assess_risk()` 之前调用，即便后来判断为 risky 写入 pending/ 甚至被人工拒绝，该回复已进入 RAG 索引，影响后续类似问题的回复质量 |
| **修改方向** | 将 Bot 回复的 RAG 索引调用移至 `risk_level == "safe"` 且 `post_comment()` 返回 `True` 之后执行，确保只有成功发布的回复才被索引 |

---

#### FIX-07：在线 embedding 场景下余弦相似度公式不准确

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.2 问题 3 |
| **涉及代码** | `scripts/rag_retriever.py:379`，`scripts/rag_retriever.py:403` |
| **原始描述** | `similarity = 1 - (dist ** 2) / 2` 仅对 L2 归一化向量成立；本地 BGE 模型使用 `normalize_embeddings=True` 故公式正确；但线上 OpenAI embedding 未归一化，导致相似度阈值（0.72）过滤失效 |
| **修改方向** | 在 `get_or_create_collection` 时设置 `metadata={"hnsw:space": "cosine"}`，并在存储在线 embedding 前对向量做 L2 归一化；或在在线 embedding 模式下改用 `1 - dist` 计算（ChromaDB cosine 空间返回的是余弦距离） |

---

#### FIX-08：向量库缓存 key 使用 `run_number`，每次运行都创建新缓存条目

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 3.1 问题 19 |
| **涉及代码** | `.github/workflows/bot.yml:66-67`，`.github/workflows/sync-wiki.yml` |
| **原始描述** | `key: vectors-${{ runner.os }}-${{ github.run_number }}` 每次递增，永远 cache miss，通过 restore-keys 恢复上次缓存，旧 key 不断积累；计划文档 AI-011 建议用日期或文件哈希作为 key |
| **修改方向** | 将 cache key 改为基于日期（如 `${{ env.date }}`）或 Wiki 文件哈希（`${{ hashFiles('csm-wiki/**') }}`），`restore-keys` 保持前缀匹配；bot.yml 和 sync-wiki.yml 同步修改 |

---

#### FIX-09：`post_comment` 吞噬 `ZhihuAuthError`，Cookie 失效在发布路径上无告警

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.1 问题 2 |
| **涉及代码** | `scripts/zhihu_client.py:392-396` |
| **原始描述** | 发布时遇到 401/403 捕获为 `False` 返回，主流程将其写入 pending/ 而不触发 `ZhihuAuthError` 的告警路径（`run_bot.py:701-705`），导致 Cookie 失效在发布路径上无声无息，与流程图 "Cookie 失效 → 创建 GitHub Issue 告警" 设计不符 |
| **修改方向** | 在 `post_comment` 中的 `ZhihuAuthError` 捕获后，在返回 `False` 之前主动记录或重新抛出，使主流程得以触发告警；或调整主流程逻辑，在 `post_comment` 返回 `False` 时检查是否有 Cookie 失效信号 |

---

#### FIX-10：`_request_with_retry` 耗尽重试时始终抛 `ZhihuRateLimitError`，误导排查

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.1 问题 1 |
| **涉及代码** | `scripts/zhihu_client.py:228` |
| **原始描述** | 循环因 `requests.RequestException`（DNS 失败、超时等）耗尽重试后，末尾统一抛 `ZhihuRateLimitError`；上层会将其解读为"知乎限流"并发出限流告警，误导排查 |
| **修改方向** | 在末尾判断 `last_exception` 的类型：若是网络类异常则抛更通用的 `ZhihuRequestError`；或在 `ZhihuRateLimitError` 中携带原始异常以便区分 |

---

#### FIX-11：`ZhihuAuthError` 告警时状态码固定为 401，403 情况误报

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.6 问题 15 |
| **涉及代码** | `scripts/run_bot.py:704` |
| **原始描述** | `self.alert_manager.alert_cookie_expired(401)` 固定传 401，但 `ZhihuAuthError` 也可能由 403（权限不足）触发，告警 Issue 中会显示错误状态码 |
| **修改方向** | 在 `ZhihuAuthError` 中添加 `status_code` 属性，从 `_request_with_retry` 传入；`run_articles` 捕获时使用 `e.status_code` 作为参数 |

---

#### FIX-12：真人回复索引时 `messages[0]` 可能是 Bot 回复而非用户提问

| 字段 | 内容 |
|------|------|
| **提出者** | codex |
| **原文位置** | `codex - 2026-04-10.md` § 主要发现 3 |
| **涉及代码** | `scripts/run_bot.py:581-589` |
| **原始描述** | 捕获作者真人回复时取最近两轮上下文并使用 `messages[0]` 作为 question；当最近两轮是"Bot 回复 → 真人回应"时，`messages[0]` 是上一轮 Bot 内容，导致索引记录的 QA 对中 question 部分被 Bot 回复占据，后续检索会召回与真实用户提问无关的文本 |
| **修改方向** | 在真人回复分支明确从 `history_messages` 中找最近一条 `role == "user"` 的内容作为 question，而非直接取 `messages[0]` |

---

### 🟢 低优先级（代码质量/可读性/文档）

---

#### FIX-13：`_parse_turns` 中 `>` 剥离正则会破坏评论中合法的 Markdown 引用

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.4 问题 7 |
| **涉及代码** | `scripts/thread_manager.py:252` |
| **原始描述** | `re.sub(r'^>\s*', '', body, flags=re.MULTILINE)` 会将用户评论内容中本身含 `>` 的行（如 Markdown 引用 "> 正如 xxx 所说"）的引用符号剥去，导致历史消息失真，影响 LLM 上下文理解 |
| **修改方向** | 改为只剥除第一层引用前缀（即仅匹配整块 `> ` 的开头行），或在写入线程时为评论内容加特殊 fence（如 `<comment>...</comment>` XML 标签），避免与 Markdown 引用格式混用 |

---

#### FIX-14：`is_followup` 参数在主流程中始终为默认值 False，追问与首评无法区分

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.4 问题 8 |
| **涉及代码** | `scripts/run_bot.py:422-427` |
| **原始描述** | `ThreadManager.append_turn` 接受 `is_followup` 参数影响线程文件中的显示格式，但主流程调用时从未设置，所有评论包括 `parent_id != None` 的追问都被记录为"评论" |
| **修改方向** | 在调用 `append_turn()` 时，根据 `comment.parent_id is not None` 判断并传入 `is_followup=True` |

---

#### FIX-15：`get_or_create_thread` 在 `_process_single_comment` 中被调用两次，第二次参数不完整

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.6 问题 13 |
| **涉及代码** | `scripts/run_bot.py:416`，`scripts/run_bot.py:469` |
| **原始描述** | 追加用户评论时调用一次，追加 Bot 回复时再调用一次；第二次 `root_comment` 只有 `id` 和 `author` 两个字段，不符合函数期望的完整格式 |
| **修改方向** | 第一次调用后将返回的 `thread_path` 保存为局部变量，后续直接复用，删除第二次 `get_or_create_thread` 调用 |

---

#### FIX-16：`max_new_comments_per_run` 配置字段定义了但从未使用

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.6 问题 14 |
| **涉及代码** | `config/settings.yaml`，`scripts/run_bot.py` |
| **原始描述** | `settings.yaml` 定义了 `bot.max_new_comments_per_run: 20`，`plan/README.md` 也提到该字段，但代码只实现了 `max_new_comments_per_day`，未实现单次运行上限；若某次有 500 条新评论，会处理到日上限而非每次运行上限 |
| **修改方向** | 在 `process_article()` 循环中增加单次运行计数器，当本次处理数达到 `max_new_comments_per_run` 时提前 `break` |

---

#### FIX-17：`daily_cost` 与 `total_cost_usd` 属性语义重复，注释相同

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.3 问题 5 |
| **涉及代码** | `scripts/llm_client.py:414-421` |
| **原始描述** | 两个属性返回同一字段 `_daily_cost_usd`，注释也相同，造成调用者混淆；实际 `run_bot.py` 使用的是 `total_cost_usd` |
| **修改方向** | 删除 `daily_cost` 属性，统一使用 `total_cost_usd`；更新测试中对 `daily_cost` 的引用 |

---

#### FIX-18：`_has_open_issue` 只搜索前 50 条 open issue，Issue 多时可能失效

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.7 问题 16 |
| **涉及代码** | `scripts/alerting.py:84` |
| **原始描述** | `params = {"state": "open", "per_page": 50}` 若仓库积累超过 50 条 open issue，新告警可能重复创建 |
| **修改方向** | 增加 `labels` 参数缩小搜索范围（只搜 `bot-alert` 标签的 issue），或实现分页逻辑遍历全部 open issue |

---

#### FIX-19：`approve_pending` done 目录路径 `archive/done/` 未在计划目录结构中定义

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.9 问题 17 |
| **涉及代码** | `scripts/archiver.py:91` |
| **原始描述** | `done_dir = filepath.parent.parent / "archive" / "done"`，`archive/` 目录已被 `ThreadManager` 用于存储对话线程（`archive/articles/{id}/threads/`），`archive/done/` 与线程文件混在同一父目录，层次不清晰；计划文档中 `archive/` 结构只设计了 `articles/` 子目录 |
| **修改方向** | 将审批完成的回复存放路径改为 `data/done/` 或独立目录 `approved/`，与线程归档目录分离；更新 `plan/README.md` 目录结构设计 |

---

#### FIX-20：`assess_risk` 未检查预算，理论上可超预算继续消耗

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.3 问题 6 |
| **涉及代码** | `scripts/llm_client.py:264-324` |
| **原始描述** | `assess_risk` 内部调用 LLM 但调用前不检查 `_daily_cost_usd >= budget_usd_per_day`；当前主流程中 `generate_reply` 抛出 `BudgetExceededError` 后不会到达 `assess_risk`，暂时安全；但未来重构顺序后存在隐患 |
| **修改方向** | 在 `assess_risk` 开头增加与 `generate_reply` 相同的预算检查逻辑 |

---

#### FIX-21：文章摘要缓存 key 使用 Python 内置 `hash()`，跨进程不一致

| 字段 | 内容 |
|------|------|
| **提出者** | claude-agent |
| **原文位置** | `claude-agent-2026-04-10.md` § 2.2.3 潜在问题 2 |
| **涉及代码** | `scripts/llm_client.py:236` |
| **原始描述** | `cache_key = f"{title}_{hash(content)}"` 使用 Python 内置 `hash()`，不同进程间哈希值不同（Python 3.3+ 启用了哈希随机化），导致 GitHub Actions 每次运行都重新生成文章摘要 |
| **修改方向** | 改为 `cache_key = f"{title}_{hashlib.md5(content.encode()).hexdigest()}"` |

---

#### FIX-22：并发安全——`bot.yml` 未设置 concurrency 限制，`seen_ids.json` 可能因并发写入损坏

| 字段 | 内容 |
|------|------|
| **提出者** | claude-agent |
| **原文位置** | `claude-agent-2026-04-10.md` § 八 高优先级改进 1 |
| **涉及代码** | `.github/workflows/bot.yml` |
| **原始描述** | `seen_ids.json` 读写无文件锁，若两次 workflow 同时运行可能导致数据损坏和重复处理 |
| **修改方向** | 在 `bot.yml` 中添加 `concurrency` 设置（`group: reply-bot`，`cancel-in-progress: false`），保证同一时间只有一个实例运行 |

---

#### FIX-23：embedding 失败时旧向量已删、新向量未入，当次运行留有空洞

| 字段 | 内容 |
|------|------|
| **提出者** | claude |
| **原文位置** | `claude-2026-04-10.md` § 2.2 问题 4 |
| **涉及代码** | `scripts/rag_retriever.py:284-290`（删除旧向量），`scripts/rag_retriever.py` embedding 步骤 |
| **原始描述** | `sync_wiki` 先删除旧向量再 embedding，若 embedding 失败（如线上 API 超限），旧向量已删、新向量未入；下次运行会重试（哈希未更新），但当次运行中该文件无法被检索 |
| **修改方向** | 将删除旧向量的步骤移到 embedding 成功并写入之后，或在失败时回滚删除操作（重新写入旧向量） |

---

## 二、文档问题汇总

| ID | 提出者 | 文档位置 | 问题描述 |
|----|--------|----------|----------|
| DOC-01 | claude | `plan/README.md` § 一（流程图） | 流程图节点 "Cookie+CSRF 评论发布是否可用？" 已被 AI 风险评估替代，流程图未更新 |
| DOC-02 | claude | `plan/README.md` § 四（settings.yaml 示例） | 示例保留了 `review.manual_mode` 字段，实际代码已删除 |
| DOC-03 | claude | `plan/README.md` AI-004 验收条件 | cron 表达式与实际 `bot.yml` 不一致（计划 6h vs 实现 15min）→ 随 FIX-03 一并修正 |
| DOC-04 | claude | `plan/README.md` AI-011 | cache key 策略与实际实现不一致 → 随 FIX-08 一并修正 |
| DOC-05 | claude | `docs/实施记录/AI-009-主流程.md` | 函数名 `write_pending_reply()` 与实际 `_write_pending()` 不符 |
| DOC-06 | claude | `docs/实施记录/AI-009-主流程.md` | 测试项数（8 项）已过时 |
| DOC-07 | claude | `docs/实施记录/AI-010-012-告警与费用.md` | 方法名 `create_alert_issue()` 与实际 `create_issue()` 不符 |

---

## 三、测试补充项

| ID | 提出者 | 问题描述 |
|----|--------|----------|
| TEST-01 | claude（问题 20） | 缺少测试验证 RAG `retrieve()` 调用时 `query` 参数为评论内容而非文章标题（对应 FIX-02） |
| TEST-02 | claude（问题 21） | 缺少测试验证 `summarize_article(content=...)` 传入的是正文而非标题（对应 FIX-05） |

---

## 四、实施顺序建议

```
第一批（正确性影响，优先修复）：
  FIX-01  question 类型读写
  FIX-02  RAG 按评论内容检索
  FIX-03  cron 频率调整
  FIX-06  Bot 回复索引时序

第二批（准确性/稳定性）：
  FIX-04  去重持久化
  FIX-05  summarize_article 内容问题
  FIX-07  在线 embedding 相似度公式
  FIX-08  缓存 key 策略
  FIX-09  AuthError 告警路径
  FIX-10  重试末尾异常类型
  FIX-11  401/403 状态码
  FIX-12  真人回复索引 question 取值
  FIX-22  并发安全

第三批（代码质量/文档）：
  FIX-13 ~ FIX-21
  DOC-01 ~ DOC-07
  TEST-01 ~ TEST-02
```
