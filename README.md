# CSM-QA-Robot

> 通用 RAG 问答 Python 库 —— 基于 CSM Wiki / LabVIEW 知识库，封装 LLM 调用与向量检索，对外仅暴露一个简洁的 `CSM_QA` 类。

本仓库经过一次大胆重构，从"知乎评论批处理机器人"演进为可被任意 Python 项目 `import` 使用的 **RAG 问答 SDK**。原有的 inbox/pending 文件流、知乎客户端、定时 Workflow 等均已移除。

---

## 安装

```bash
pip install -e .
# 或
pip install -r requirements.txt
```

依赖：`openai>=1.0`、`chromadb>=0.4`、`sentence-transformers>=2.2`、`charset-normalizer>=3.0`。

---

## 60 秒上手

```python
from csm_qa import CSM_QA

qa = CSM_QA(api_key="sk-deepseek-xxx")           # 默认 provider=deepseek
answer = qa.ask("CSM 框架中的状态机如何切换？")
print(answer)
```

带历史对话：

```python
from csm_qa import CSM_QA, Message

qa = CSM_QA(api_key="sk-xxx")
history = [
    Message(role="user", content="CSM 是什么？"),
    Message(role="assistant", content="CSM 是 Communicable State Machine ..."),
]
answer = qa.ask("那它和 JKI SM 的区别？", history=history)
```

需要更多元信息？

```python
result = qa.ask_detailed("CSM 状态机如何切换？", history=history)
print(result.answer)         # 文本
print(result.contexts)       # 命中的 RAG 片段
print(result.usage)          # token 用量
print(result.prompt_messages)  # 实际发往 LLM 的 messages（调试用）
```

---

## API 参数

```python
CSM_QA(
    api_key,                                # 必填
    *,
    provider="deepseek",                    # "deepseek" 或 "openai_compatible"
    model=None,                             # None → 取 provider 默认
    base_url=None,                          # None → 取 provider 默认
    temperature=0.5,
    max_tokens=512,
    max_retries=3,
    request_timeout=60.0,

    wiki_dir="csm-wiki/remote",             # 知识库目录
    vector_store_dir=".csm_qa/vector_store",
    embedding_provider="local",             # "local"（本地）或 "openai"
    embedding_model="BAAI/bge-small-zh-v1.5",
    embedding_api_key=None,
    embedding_base_url=None,
    top_k=3,
    similarity_threshold=0.72,

    system_prompt=None,                     # None → 内置 CSM/LabVIEW prompt
    auto_sync_wiki=True,                    # 首次运行若向量库为空，自动同步
)
```

### 支持的 LLM 供应商

| `provider`            | 默认 `base_url`                    | 默认 `model`     | 备注                              |
| --------------------- | ---------------------------------- | ---------------- | --------------------------------- |
| `deepseek`            | `https://api.deepseek.com`         | `deepseek-chat`  | DeepSeek 官方                      |
| `openai_compatible`   | 必须传                             | 必须传           | OpenAI 官方、Moonshot、智谱、本地 vLLM/Ollama 等任意 OpenAI 兼容服务 |

示例：使用 OpenAI 官方

```python
qa = CSM_QA(
    api_key="sk-xxx",
    provider="openai_compatible",
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
)
```

### 从环境变量构造

```python
qa = CSM_QA.from_env()
# 兼容旧变量：LLM_API_KEY / LLM_MODEL / LLM_BASE_URL
# 新变量：CSM_QA_API_KEY / CSM_QA_PROVIDER / CSM_QA_MODEL / CSM_QA_BASE_URL
```

---

## 知识库

把任意 Markdown 文档放入 `csm-wiki/remote/` 目录即可（支持子目录、UTF-8 / GBK / Big5 自动识别）。

- 首次构造 `CSM_QA` 时若向量库为空：
  - 若 `csm-wiki/remote/` 目录不存在但 `csm-wiki/wiki_source.json` 存在，会自动从远程仓库克隆 wiki 并建立索引。
  - 若 `csm-wiki/remote/` 已存在，则直接对目录做增量同步。
  - 若两者均不存在，则跳过同步（无 RAG 上下文，仅凭 LLM 本身回答）。
- 之后可通过命令行手动增量同步：

```bash
python -m csm_qa.sync_wiki                 # 增量
python -m csm_qa.sync_wiki --force         # 强制重建
python -m csm_qa.sync_wiki --wiki ./docs --store ./.csm_qa/vector_store

# 通过 wiki_source.json 检查远程更新并按需拉取
python -m csm_qa.sync_wiki --remote
```

或在代码中：

```python
qa.sync_wiki(force=False)
```

---

## 提示词

库内置一段针对 **CSM/LabVIEW + RAG** 的中文 system prompt（详见 [`csm_qa/prompts.py`](csm_qa/prompts.py) 的 `DEFAULT_SYSTEM_PROMPT`）。

如需替换为通用领域 / 英文 / 自定义风格，传入 `system_prompt=` 即可：

```python
qa = CSM_QA(api_key="sk", system_prompt="You are a helpful general-purpose assistant.")
```

---

## 项目结构

```
.
├── csm_qa/                 # SDK 主包（pip install -e . 后可 import）
│   ├── __init__.py         # 导出 CSM_QA / Message / AnswerResult
│   ├── api.py              # CSM_QA 主类
│   ├── llm.py              # OpenAI 兼容 LLM 客户端
│   ├── rag.py              # ChromaDB + Embedding 检索器
│   ├── providers.py        # provider 预设（deepseek / openai_compatible）
│   ├── prompts.py          # 默认 system prompt
│   ├── types.py            # Message / AnswerResult / Usage
│   └── sync_wiki.py        # CLI: python -m csm_qa.sync_wiki
├── csm-wiki/               # 默认知识库目录（放置 .md 文档）
├── examples/
│   ├── basic_usage.py
│   └── multi_turn.py
├── tests/                  # 单元测试
├── docs/                   # 历史调研与设计文档（仅参考，非运行时依赖）
├── pyproject.toml
└── requirements.txt
```

---

## 测试

```bash
pip install -e .[test]
python -m pytest tests/ -v
```

测试用 mock OpenAI 客户端 + 词袋式 fake embedding，无需真实 API key 与模型下载。

---

## License

MIT
