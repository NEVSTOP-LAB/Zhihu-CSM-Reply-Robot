# -*- coding: utf-8 -*-
"""
RAG 检索器 — ChromaDB + BGE Embedding
======================================

实施计划关联：AI-005 RAGRetriever — Wiki 索引与检索
参考文档：docs/调研/04-CSM-Wiki-RAG知识库.md

功能：
- CSM Wiki 文档增量 embedding（MD5 比对）
- 混合检索（真人回复优先 + Wiki 补充）
- 本地/线上 embedding 双模式支持
"""
