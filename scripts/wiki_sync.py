# -*- coding: utf-8 -*-
"""
CSM Wiki 增量同步模块
======================

实施计划关联：docs/plan/README.md sync-wiki.yml
参考文档：docs/调研/04-CSM-Wiki-RAG知识库.md

功能：
- CSM Wiki 文档变更检测（MD5 哈希比对）
- 增量更新向量库
- 支持强制重建

使用方式：
    python scripts/wiki_sync.py
    FORCE_REBUILD=true python scripts/wiki_sync.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rag_retriever import RAGRetriever

logger = logging.getLogger(__name__)


def main():
    """Wiki 同步入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).parent.parent

    # 加载配置
    settings_path = project_root / "config" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    rag_cfg = settings.get("rag", {})

    # 初始化 RAGRetriever
    retriever = RAGRetriever(
        wiki_dir=str(project_root / "csm-wiki"),
        vector_store_dir=str(project_root / "data" / "vector_store"),
        reply_index_dir=str(project_root / "data" / "reply_index"),
        use_online_embedding=rag_cfg.get("use_online_embedding", False),
        embedding_model=rag_cfg.get(
            "embedding_model", "BAAI/bge-small-zh-v1.5"
        ),
    )

    # 检查是否强制重建
    force = os.environ.get("FORCE_REBUILD", "").lower() in ("true", "1", "yes")

    logger.info(f"开始 Wiki 同步 (force={force})")
    retriever.sync_wiki(force=force)
    logger.info("Wiki 同步完成")


if __name__ == "__main__":
    main()
