# 调研总览

本目录汇总了构建 **csm_qa RAG 问答 SDK** 所需的各技术方向调研报告。

## 整体架构

```
用户调用 CSM_QA.ask(question)
    │
    ▼
RAG 检索 CSM Wiki（ChromaDB 向量检索）
    │  返回 top-K 相关文档片段
    ▼
组装 Prompt
    │  System: 内置 CSM/LabVIEW 角色 + RAG 片段
    │  User: 当前问题 + 可选对话历史
    ▼
LLM 生成回答（OpenAI 兼容接口）
    │  默认 provider: deepseek-chat
    ▼
返回 AnswerResult（answer / contexts / usage）
```

## 调研文档列表

| 文档 | 内容 |
|------|------|
| [03-LLM接入与回复生成.md](./03-LLM接入与回复生成.md) | DeepSeek/OpenAI 模型对比、OpenAI 兼容接口、Prompt 设计 |
| [04-CSM-Wiki-RAG知识库.md](./04-CSM-Wiki-RAG知识库.md) | RAG vs Skill 对比、ChromaDB 向量库、BGE Embedding、增量更新 |
| [06-Token优化策略.md](./06-Token优化策略.md) | DeepSeek Prompt Caching、RAG 精准检索、批量处理优化 |
| [07-费用评估.md](./07-费用评估.md) | SDK 费用测算、DeepSeek vs OpenAI 成本对比、敏感性分析 |
