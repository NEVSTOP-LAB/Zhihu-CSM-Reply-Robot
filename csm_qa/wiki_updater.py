"""Wiki 远程同步：通过 GitHub API 检查最新 commit，按需 clone/pull 并更新 RAG。

用法示例::

    from csm_qa.wiki_updater import check_and_update_wiki
    from csm_qa.rag import EmbeddingFunction, RAGRetriever

    embedding_fn = EmbeddingFunction()
    retriever = RAGRetriever(
        wiki_dir="csm-wiki/remote",
        vector_store_dir=".csm_qa/vector_store",
        embedding_fn=embedding_fn,
    )
    updated = check_and_update_wiki(
        source_file="csm-wiki/wiki_source.json",
        local_dir="csm-wiki/remote",
        retriever=retriever,
    )
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WikiSource:
    """wiki 源信息：远程仓库 URL 与上次同步的 commit ID。"""

    url: str
    commit_id: str

    @classmethod
    def load(cls, source_file: str | Path) -> "WikiSource":
        """从 JSON 文件读取 :class:`WikiSource`。"""
        with open(source_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(url=data["url"], commit_id=data.get("commit_id", ""))

    def save(self, source_file: str | Path) -> None:
        """将当前状态写回 JSON 文件。"""
        with open(source_file, "w", encoding="utf-8") as f:
            json.dump(
                {"url": self.url, "commit_id": self.commit_id},
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")


def _repo_api_url(repo_url: str) -> str:
    """将 GitHub 仓库 URL 转换为 API URL 前缀。

    支持 ``https://github.com/owner/repo`` 和
    ``https://github.com/owner/repo.git`` 两种格式。
    """
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        repo_url.strip(),
    )
    if not m:
        raise ValueError(f"不支持的 GitHub 仓库 URL 格式: {repo_url!r}")
    owner, repo = m.group(1), m.group(2)
    return f"https://api.github.com/repos/{owner}/{repo}"


def fetch_latest_commit_id(
    repo_url: str,
    branch: str = "main",
    timeout: float = 15.0,
) -> str:
    """通过 GitHub API 获取指定分支的最新 commit SHA。

    Args:
        repo_url: GitHub 仓库 URL，例如
            ``https://github.com/NEVSTOP-LAB/CSM-Wiki``。
        branch: 分支名，默认 ``"main"``。若 ``main`` 不存在会自动回退到
            ``master``。
        timeout: HTTP 请求超时秒数。

    Returns:
        完整 40 位 commit SHA 字符串。

    Raises:
        urllib.error.URLError: 网络请求失败。
        ValueError: API 返回格式意外。
    """
    api_base = _repo_api_url(repo_url)
    branches_to_try = [branch] if branch != "main" else ["main", "master"]

    last_exc: Optional[Exception] = None
    for br in branches_to_try:
        url = f"{api_base}/commits/{br}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "csm-qa-wiki-updater/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                sha = data.get("sha", "")
                if sha:
                    return sha
                raise ValueError(f"GitHub API 返回中缺少 sha 字段: {url}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                last_exc = exc
                continue
            raise
    raise ValueError(
        f"无法获取 {repo_url!r} 的最新 commit（尝试过: {branches_to_try}）"
    ) from last_exc


def pull_wiki(repo_url: str, local_dir: str | Path) -> None:
    """Clone（首次）或 pull（已存在）wiki 仓库到本地目录。

    Args:
        repo_url: GitHub 仓库 URL。
        local_dir: 本地目标目录；不存在时自动 clone，已存在时执行 pull。
    """
    local_path = Path(local_dir)
    git_dir = local_path / ".git"

    if not local_path.exists() or not git_dir.exists():
        logger.info("克隆 wiki: %s → %s", repo_url, local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", repo_url, str(local_path)],
            check=True,
            capture_output=True,
        )
    else:
        logger.info("拉取 wiki 更新: %s", local_path)
        subprocess.run(
            ["git", "-C", str(local_path), "pull", "--ff-only"],
            check=True,
            capture_output=True,
        )


def check_and_update_wiki(
    source_file: str | Path,
    local_dir: str | Path,
    retriever,
    branch: str = "main",
    force_sync: bool = False,
    timeout: float = 15.0,
) -> bool:
    """检查远程 wiki 是否有更新，有则拉取并重建 RAG。

    Args:
        source_file: ``wiki_source.json`` 路径。
        local_dir: wiki 本地克隆目录。
        retriever: :class:`~csm_qa.rag.RAGRetriever` 实例，用于触发 RAG 同步。
        branch: 远程分支名。
        force_sync: 即使 commit 未变化也强制重建 RAG。
        timeout: GitHub API 请求超时秒数。

    Returns:
        ``True`` 表示进行了拉取并更新了 RAG；``False`` 表示已是最新，无需操作。
    """
    source_file = Path(source_file)
    wiki_src = WikiSource.load(source_file)

    logger.info("检查 wiki 远程更新: %s", wiki_src.url)
    latest = fetch_latest_commit_id(wiki_src.url, branch=branch, timeout=timeout)
    local_short = wiki_src.commit_id[:12] if wiki_src.commit_id else "(空)"
    logger.info("远程最新 commit: %s  本地已有: %s", latest[:12], local_short)

    if not force_sync and latest == wiki_src.commit_id:
        logger.info("wiki 已是最新，跳过拉取与 RAG 更新")
        return False

    pull_wiki(wiki_src.url, local_dir)

    logger.info("触发 RAG 同步（force=%s）", force_sync)
    result = retriever.sync_wiki(force=force_sync)
    logger.info("RAG 同步完成: %s", result)

    wiki_src.commit_id = latest
    wiki_src.save(source_file)
    logger.info("wiki_source.json 已更新，commit_id=%s", latest[:12])
    return True
