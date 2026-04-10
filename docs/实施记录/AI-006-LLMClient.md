# AI-006 实施记录：LLMClient — 回复生成

## 任务目标
实现 DeepSeek/OpenAI LLM 调用封装，含 Prompt Caching 和费用追踪。

## 实施内容

### 1. 模块实现 (`scripts/llm_client.py`)
- **LLMClient**: OpenAI 兼容接口封装，通过 base_url 切换 provider
- **generate_reply()**: 
  - System Prompt 固定前缀（角色+规则），最大化 Prompt Caching 命中
  - Wiki 片段紧随 System Prompt，同文章批量时缓存命中
  - history_messages 正确拼接（追问场景）
  - 当前评论放 User 消息，不破坏缓存前缀
- **summarize_article()**: 文章摘要生成，内存缓存避免重复调用
- **_call_with_retry()**: 指数退避重试最多3次（RateLimitError + 500 错误）
- **费用追踪**: 基于 DeepSeek 定价，支持缓存命中折扣计算
- **BudgetExceededError**: 超过 llm_budget_usd_per_day 时抛出

### 2. 设计决策
- SYSTEM_PROMPT_PREFIX 作为模块常量，确保缓存一致性
- PRICING 字典支持多模型定价，包含缓存命中折扣
- 文章摘要使用 hash(content) 作为缓存键

## 测试结果
```
tests/test_llm_client.py — 15 项测试全部通过 ✅
- TestSystemPrompt: 3 项（固定前缀、Wiki片段、文章摘要在System中）
- TestHistoryMessages: 2 项（历史消息拼接、无历史时结构正确）
- TestRetryLogic: 2 项（限流重试成功、重试耗尽抛出异常）
- TestBudgetControl: 3 项（超预算抛出、费用累计、费用重置）
- TestSummarizeArticle: 3 项（返回摘要、缓存命中、不同文章不共享）
- TestReturnValues: 2 项（返回元组、stats属性）
```

## 验收状态
✅ 单元测试全部通过
