# AI-006 实施记录：LLMClient — 回复生成

## 状态：✅ 完成

## 实施内容

### LLMClient 类实现 (`scripts/llm_client.py`)
- OpenAI 兼容接口（支持 DeepSeek）
- `generate_reply(comment, context_chunks, article_summary, history_messages)` → `(reply, tokens)`
  - 固定 System Prompt 前缀（角色+规则），最大化 Prompt Caching 命中
  - RAG context 和文章摘要注入 System Prompt
  - history_messages 正确拼接（追问场景）
- `summarize_article(title, content)` → 摘要文本
  - 结果缓存：相同文章不重复调用 API
- 指数退避重试最多 3 次（RateLimitError / APIError）
- 费用追踪：自动累计 prompt_tokens, completion_tokens, cache_hit_tokens
- 预算控制：超 budget_usd_per_day 时抛 BudgetExceededError
- PRICING 字典：DeepSeek-V3/R1 价格

## 测试结果
```
16 passed in 1.20s
```

覆盖：System Prompt（3）、消息构建（3）、重试逻辑（2）、预算控制（4）、文章摘要缓存（3）、返回值（1）

## 验收标准
- [x] System Prompt 前缀固定（缓存友好）
- [x] history_messages 正确拼接
- [x] 重试逻辑正确
- [x] 超预算时 BudgetExceededError
- [x] summarize_article 缓存（第二次不触发 API）
