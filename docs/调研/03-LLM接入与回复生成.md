# LLM 接入与回复生成调研

> **调研结论摘要**
>
> 推荐使用 **DeepSeek-V3**（`deepseek-chat`）作为主力模型：中文质量优秀、价格比 GPT-4o-mini 低约 50%、与 OpenAI SDK 完全兼容，只需切换 `base_url`。
> 所有主流国产/海外模型均支持 OpenAI 兼容接口，通过 `provider` 参数可零代码切换。
> 对复杂技术问题保留 GPT-4o 或 DeepSeek-R1 作备选。
> Prompt 设计：固定部分前置（触发 Prompt Caching），用户问题放 User 消息。

## 1. API 兼容性

SDK 使用 **OpenAI 兼容接口**（Chat Completions），支持接入：
- OpenAI（GPT-4o, GPT-4o-mini 等）
- **DeepSeek**（deepseek-chat, deepseek-reasoner）— 推荐主力
- Azure OpenAI
- 任意兼容 `/v1/chat/completions` 的自托管或第三方模型

通过 `provider` / `base_url` / `api_key` 参数切换 provider，无需改代码。

```python
from openai import OpenAI

client = OpenAI(
    api_key=api_key,
    base_url=base_url,   # 默认 "https://api.deepseek.com"
)
```

参考：[openai-python SDK](https://github.com/openai/openai-python) | [DeepSeek API Docs](https://api-docs.deepseek.com/)

## 2. 回答生成流程

```
用户问题 → [RAG 检索相关 CSM 片段] → 组装 Prompt → LLM → 生成回答文本
```

### Prompt 结构（三段式）

```
System（静态，适合 Prompt Caching）:
  你是 CSM/LabVIEW 技术助理，帮助用户解答 CSM 框架相关问题。
  回复风格：专业、友善、简洁。
  [CSM Wiki 相关片段] ← RAG 动态注入，放在 System 尾部

User（每次不同）:
  用户问题（+ 可选多轮对话历史）
```

> 将 System Prompt 的**固定部分**放在最前，RAG 片段紧随其后；动态内容（用户问题）放 User，最大化缓存命中率。

## 3. 模型选型对比

### 3.1 主流模型对比

| 模型 | Provider | 输入 ($/1M tokens) | 输出 ($/1M tokens) | 中文质量 | 推荐场景 |
|------|----------|--------------------|--------------------|----------|----------|
| **deepseek-chat (V3)** | DeepSeek | $0.07（缓存命中）/ $0.27 | $1.10 | ⭐⭐⭐⭐⭐ | **日常问答（首选）** |
| deepseek-reasoner (R1) | DeepSeek | $0.14（缓存命中）/ $0.55 | $2.19 | ⭐⭐⭐⭐⭐ | 复杂推理/技术问题 |
| gpt-4o-mini | OpenAI | $0.15 | $0.60 | ⭐⭐⭐⭐ | 备选，中文略弱 |
| gpt-4o | OpenAI | $2.50 | $10.00 | ⭐⭐⭐⭐⭐ | 高质量（费用较高） |
| 本地模型（Ollama/vLLM） | 自托管 | $0 | $0 | 视模型而定 | 私有化部署 |

> 价格参考：[DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing/) | [OpenAI Pricing](https://openai.com/pricing)

### 3.2 DeepSeek 接入示例

DeepSeek API 与 OpenAI SDK 完全兼容，仅需切换 `base_url`：

```python
# DeepSeek（推荐）
client = OpenAI(
    api_key=api_key,                         # DEEPSEEK_API_KEY
    base_url="https://api.deepseek.com",
)
model = "deepseek-chat"     # 或 "deepseek-reasoner" 用于复杂问题

# OpenAI（备选）
# base_url="https://api.openai.com/v1", model="gpt-4o-mini"
```

### 3.3 推荐方案

**默认：DeepSeek-V3（deepseek-chat）**
- 中文能力与 GPT-4o 相当，价格约为 gpt-4o-mini 的 50%
- DeepSeek 原生支持 Prompt Caching（缓存命中时输入费用降至 $0.07/1M）
- 国内访问延迟低

**升级路径：**
- 简单问答 → `deepseek-chat`（低成本）
- 复杂技术问题 → `deepseek-reasoner` 或 `gpt-4o`（质量优先）

## 4. 调用示例

```python
def generate_answer(question: str, context_chunks: list[str]) -> str:
    wiki_context = "\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 CSM/LabVIEW 技术助理，帮助用户解答 CSM 框架相关问题。"
                    "回复风格：专业、友善、简洁。\n\n"
                    f"参考知识库：\n{wiki_context}"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.5,
        max_tokens=512,
    )
    return response.choices[0].message.content
```

## 5. 回答质量控制

- **温度**：0.5，平衡一致性与自然度
- **max_tokens**：512（可通过 `max_tokens=` 参数调整）
- **重试机制**：内置指数退避重试（默认 3 次）

## 6. 错误处理

```python
import time
from openai import RateLimitError, APIError

def call_with_retry(fn, max_retries=3):
    for i in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            time.sleep(2 ** i)
        except APIError as e:
            if e.status_code >= 500:
                time.sleep(2 ** i)
            else:
                raise
    raise RuntimeError("LLM call failed after retries")
```

## 7. 参考资源

- [OpenAI Python SDK 文档](https://github.com/openai/openai-python)
- [OpenAI API Reference - Chat Completions](https://platform.openai.com/docs/api-reference/chat)
- [DeepSeek API Docs](https://api-docs.deepseek.com/)
- [DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [Prompt Caching - OpenAI](https://openai.com/index/api-prompt-caching/)
- [DeepSeek Prompt Caching](https://api-docs.deepseek.com/guides/kv_cache)
