# AI-009 实施记录：主流程 run_bot.py — MVP 版

## 状态：✅ 完成

## 实施内容

### BotRunner 类实现 (`scripts/run_bot.py`)
- `load_config()` — 加载 settings.yaml + articles.yaml
- `init_modules()` — 初始化所有子模块（ZhihuClient, RAG, LLM, ThreadManager, Filter, AlertManager, CostTracker）
- `load_seen_ids() / save_seen_ids()` — 已处理评论 ID 持久化
- `run_articles()` — 遍历文章，捕获异常触发告警
- `process_article(article)` — 单篇文章处理
  - 拉取评论 → 过滤已处理 → 文章摘要 → RAG 检索 → 逐条处理
- `_process_single_comment(...)` — 单条评论处理
  - 真人回复 → 索引（AI-013）
  - 前置过滤 → 截断
  - 构建线程 → 历史上下文
  - LLM 生成回复 → 写入 pending/
  - 预算超限 → raise
  - 连续失败 → 告警
- `_handle_human_reply(...)` — 真人回复索引
- `_write_pending(...)` — 写入 pending/ 待审核文件
- `run()` — 完整主流程（配置→初始化→Wiki同步→处理→保存→报告）
- `main()` — 入口函数

### 主要设计决策
- `manual_mode=True` 时写 pending/，`ZHIHU_AUTO_POST=true` 时自动发布
- 预算告警只在 `run_articles()` 层触发，避免重复
- 连续失败 ≥ consecutive_fail_limit 时告警并暂停

## 测试结果
```
14 passed in 1.69s
```

### 遇到的问题
- `run_articles()` 方法最初不存在，测试调用失败 → 添加了独立的 `run_articles()` 方法
- BudgetExceededError 在 `process_article` 和 `run_articles` 中重复触发告警 → 修改 `process_article` 只 raise，`run_articles` 负责告警

覆盖：配置加载（1）、seen_ids（2）、文章处理（5）、异常处理（2）、每日上限（2）、pending（2）

## 验收标准
- [x] 新评论生成回复并写入 pending/
- [x] 已处理评论被跳过
- [x] 广告评论被过滤
- [x] 真人回复被索引
- [x] ZhihuAuthError 触发告警
- [x] BudgetExceededError 触发告警
- [x] seen_ids 正确持久化
