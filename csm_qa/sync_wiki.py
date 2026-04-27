"""命令行入口：``python -m csm_qa.sync_wiki``。

直接构建 :class:`~csm_qa.rag.RAGRetriever` 并触发同步，无需 LLM API key。

用法::

    python -m csm_qa.sync_wiki                   # 增量同步默认目录
    python -m csm_qa.sync_wiki --wiki ./csm-wiki/remote --store ./.csm_qa/vector_store
    python -m csm_qa.sync_wiki --force           # 强制重建

    # 远程模式：通过 wiki_source.json 检查最新 commit，按需拉取并更新 RAG
    python -m csm_qa.sync_wiki --remote
    python -m csm_qa.sync_wiki --remote --source csm-wiki/wiki_source.json --wiki csm-wiki/remote
"""

from __future__ import annotations

import argparse
import logging
import sys

from csm_qa.rag import EmbeddingFunction, RAGRetriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync CSM wiki vector store.")
    parser.add_argument(
        "--wiki", default="csm-wiki/remote", help="wiki 本地目录"
    )
    parser.add_argument(
        "--store", default=".csm_qa/vector_store", help="向量库目录"
    )
    parser.add_argument(
        "--embedding-provider",
        default="local",
        choices=["local", "openai"],
    )
    parser.add_argument(
        "--embedding-model", default="BAAI/bge-small-zh-v1.5"
    )
    parser.add_argument("--force", action="store_true", help="强制重建")

    # 远程模式
    parser.add_argument(
        "--remote",
        action="store_true",
        help="远程模式：通过 wiki_source.json 检查远程 commit，有更新则拉取并重建 RAG",
    )
    parser.add_argument(
        "--source",
        default="csm-wiki/wiki_source.json",
        help="wiki_source.json 路径（仅 --remote 模式使用）",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="远程分支名（仅 --remote 模式使用）",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    embedding_fn = EmbeddingFunction(
        provider=args.embedding_provider,
        model=args.embedding_model,
    )
    retriever = RAGRetriever(
        wiki_dir=args.wiki,
        vector_store_dir=args.store,
        embedding_fn=embedding_fn,
    )

    if args.remote:
        from csm_qa.wiki_updater import check_and_update_wiki

        updated = check_and_update_wiki(
            source_file=args.source,
            local_dir=args.wiki,
            retriever=retriever,
            branch=args.branch,
            force_sync=args.force,
        )
        print("wiki_updated=" + str(updated).lower())
        return 0

    result = retriever.sync_wiki(force=args.force)
    print(
        f"updated={result['updated']} "
        f"skipped={result['skipped']} "
        f"removed={result['removed']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
