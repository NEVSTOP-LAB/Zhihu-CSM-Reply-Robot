# -*- coding: utf-8 -*-
"""
知乎 CSM 自动回复机器人 - 脚本模块
===================================

本包包含机器人的核心功能模块，按照实施计划（docs/plan/README.md）组织：

模块列表：
- zhihu_client: 知乎 API v4 封装（AI-003）
- rag_retriever: ChromaDB + BGE embedding 检索（AI-005）
- llm_client: DeepSeek/OpenAI 调用封装（AI-006）
- thread_manager: 对话线程管理（AI-007）
- comment_filter: 评论前置过滤（AI-008）
- run_bot: 主入口（AI-009）
- alerting: GitHub Issue 告警（AI-010）
- cost_tracker: 费用监控（AI-012）
- wiki_sync: CSM Wiki 增量同步
- archiver: 归档写入
"""
