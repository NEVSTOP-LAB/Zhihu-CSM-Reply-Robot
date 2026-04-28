#!/usr/bin/env python3
"""CSM Q&A Discussion Bot.

从 GitHub Discussions 中的 Q&A category 读取新问题，通过 CSM_QA SDK 生成回答后发布评论。

支持两种触发模式
──────────────────
event 模式（由 GitHub Actions discussion 事件触发）：
    python scripts/discussion_bot.py --event

manual 模式（手动触发或 workflow_dispatch）：
    python scripts/discussion_bot.py --discussion-number 42
    python scripts/discussion_bot.py --discussion-number 42 --dry-run

环境变量
──────────────────
CSM_QA_API_KEY   Fine-grained PAT（需要 repository discussions: read & write 权限）
LLM_API_KEY      DeepSeek / 其他 LLM 的 API Key
GITHUB_REPOSITORY  格式 owner/repo，如 NEVSTOP-LAB/CSM-QA-Robot（Actions 自动注入）
GITHUB_EVENT_PATH  event 模式时由 Actions 自动注入
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

# ── 确保包根目录在 sys.path（直接运行 scripts/ 时使用）──────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from csm_qa import CSM_QA  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("discussion_bot")

# ── 常量 ────────────────────────────────────────────────────────────────────

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
# Q&A Category 名称（大小写精确匹配）
QA_CATEGORY_NAME = "Q&A"
# 追加在 Bot 回复末尾的 HTML 注释，用于防重复检测（用户不可见）
BOT_MARKER = "<!-- csm-qa-bot -->"
# Bot 回复页脚（可见文字）
BOT_FOOTER = (
    "\n\n---\n"
    "> 🤖 此回答由 [CSM-QA-Robot](https://github.com/NEVSTOP-LAB/CSM-QA-Robot) 自动生成。"
    "如有偏差，欢迎追问或修正。"
)


# ── GitHub GraphQL 客户端 ───────────────────────────────────────────────────


class GitHubGraphQL:
    """轻量级 GitHub GraphQL 客户端（仅依赖 stdlib urllib）。"""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("GitHub token (CSM_QA_API_KEY) 未配置")
        self._token = token

    def query(self, gql: str, variables: Optional[dict] = None) -> dict:
        """执行 GraphQL 查询，返回 ``data`` 节点。

        Raises:
            RuntimeError: HTTP 错误或 GraphQL errors 字段存在。
        """
        payload = json.dumps({"query": gql, "variables": variables or {}}).encode()
        req = urllib.request.Request(
            GITHUB_GRAPHQL_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "csm-qa-bot/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub GraphQL HTTP {exc.code}: {body[:400]}"
            ) from exc

        result: dict = json.loads(raw)
        if result.get("errors"):
            messages = "; ".join(e.get("message", "") for e in result["errors"])
            raise RuntimeError(f"GitHub GraphQL errors: {messages}")
        return result.get("data", {})


# ── 业务逻辑 ────────────────────────────────────────────────────────────────


def _get_repo_parts() -> tuple[str, str]:
    """从 GITHUB_REPOSITORY 环境变量解析 owner/repo。"""
    repo_env = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo_env:
        raise ValueError(
            f"GITHUB_REPOSITORY 格式不正确: {repo_env!r}，期望 'owner/repo'"
        )
    owner, repo = repo_env.split("/", 1)
    return owner, repo


def resolve_qa_category_id(client: GitHubGraphQL, owner: str, repo: str) -> str:
    """通过 GraphQL 查询仓库中名为 'Q&A' 的 Discussion Category node ID。

    Raises:
        RuntimeError: 未找到 Q&A category。
    """
    gql = """
    query($owner: String!, $repo: String!) {
      repository(owner: $owner, name: $repo) {
        discussionCategories(first: 30) {
          nodes {
            id
            name
            slug
            isAnswerable
          }
        }
      }
    }
    """
    data = client.query(gql, {"owner": owner, "repo": repo})
    nodes = (
        data.get("repository", {})
        .get("discussionCategories", {})
        .get("nodes", [])
    )
    for node in nodes:
        if node.get("name", "").strip() == QA_CATEGORY_NAME:
            logger.info(
                "找到 Q&A category: id=%s slug=%s isAnswerable=%s",
                node["id"],
                node.get("slug"),
                node.get("isAnswerable"),
            )
            return node["id"]
    available = [n.get("name") for n in nodes]
    raise RuntimeError(
        f"未找到名为 {QA_CATEGORY_NAME!r} 的 Discussion Category，"
        f"仓库已有分类: {available}"
    )


def fetch_discussion(
    client: GitHubGraphQL, owner: str, repo: str, number: int
) -> dict[str, Any]:
    """拉取指定 discussion 的详情（含前 100 条评论）。

    注意：每次最多拉取 100 条评论。若 discussion 评论数超过 100，
    超出部分不会被检查，可能导致已有 Bot 回复未被检测到（重复回复）。
    """
    gql = """
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        discussion(number: $number) {
          id
          number
          title
          body
          url
          category {
            id
            name
          }
          comments(first: 100) {
            nodes {
              id
              body
              author { login }
            }
          }
        }
      }
    }
    """
    data = client.query(gql, {"owner": owner, "repo": repo, "number": number})
    disc = data.get("repository", {}).get("discussion")
    if not disc:
        raise RuntimeError(f"Discussion #{number} 不存在或无权限访问")
    return disc


def has_bot_replied(discussion: dict) -> bool:
    """检查 Discussion 评论中是否已含 BOT_MARKER，防止重复回复。"""
    for comment in discussion.get("comments", {}).get("nodes", []):
        if BOT_MARKER in (comment.get("body") or ""):
            return True
    return False


def post_comment(client: GitHubGraphQL, discussion_id: str, body: str) -> str:
    """向 Discussion 发布评论，返回新评论的 node ID。"""
    gql = """
    mutation($discussionId: ID!, $body: String!) {
      addDiscussionComment(input: {discussionId: $discussionId, body: $body}) {
        comment {
          id
          url
        }
      }
    }
    """
    data = client.query(gql, {"discussionId": discussion_id, "body": body})
    comment = data.get("addDiscussionComment", {}).get("comment", {})
    comment_url = comment.get("url", "")
    logger.info("评论已发布: %s", comment_url)
    return comment.get("id", "")


def build_reply(answer: str) -> str:
    """拼装最终回复正文（答案 + 页脚 + 防重标记）。"""
    return f"{answer.rstrip()}{BOT_FOOTER}\n{BOT_MARKER}"


def process_discussion(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    owner: str,
    repo: str,
    number: int,
    qa_category_id: str,
    *,
    dry_run: bool = False,
) -> bool:
    """处理单条 Discussion，返回 True 表示已回复（或 dry-run 打印），False 表示跳过。"""
    discussion = fetch_discussion(client, owner, repo, number)
    disc_category_id = discussion.get("category", {}).get("id", "")
    disc_category_name = discussion.get("category", {}).get("name", "")

    if disc_category_id != qa_category_id:
        logger.info(
            "Discussion #%d 分类为 %r，非 Q&A，跳过",
            number,
            disc_category_name,
        )
        return False

    if has_bot_replied(discussion):
        logger.info("Discussion #%d 已有 Bot 回复，跳过", number)
        return False

    title = discussion.get("title", "").strip()
    body = discussion.get("body", "").strip()
    question = f"{title}\n\n{body}" if body else title
    disc_id = discussion["id"]
    disc_url = discussion.get("url", f"#{number}")

    logger.info("正在为 Discussion #%d 生成回答: %r", number, title)
    answer = qa_engine.ask(question)
    reply_body = build_reply(answer)

    if dry_run:
        logger.info("=== [DRY-RUN] Discussion #%d (%s) ===", number, disc_url)
        logger.info("--- 回复内容 ---\n%s", reply_body)
        return True

    post_comment(client, disc_id, reply_body)
    logger.info("Discussion #%d 已成功回复", number)
    return True


# ── 触发模式处理 ─────────────────────────────────────────────────────────────


def run_event_mode(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    owner: str,
    repo: str,
    qa_category_id: str,
    *,
    dry_run: bool = False,
) -> None:
    """读取 GITHUB_EVENT_PATH 中的 payload，处理 discussion.created 事件。"""
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not os.path.isfile(event_path):
        raise RuntimeError(
            f"GITHUB_EVENT_PATH 未设置或文件不存在: {event_path!r}"
        )

    with open(event_path, encoding="utf-8") as f:
        event: dict = json.load(f)

    action = event.get("action", "")
    if action not in ("created",):
        logger.info("事件 action=%r，不处理（仅处理 created）", action)
        return

    discussion = event.get("discussion", {})
    number = discussion.get("number")
    if not number:
        raise RuntimeError("event payload 中缺少 discussion.number")

    process_discussion(
        client,
        qa_engine,
        owner,
        repo,
        int(number),
        qa_category_id,
        dry_run=dry_run,
    )


def run_manual_mode(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    owner: str,
    repo: str,
    qa_category_id: str,
    discussion_number: int,
    *,
    dry_run: bool = False,
) -> None:
    """手动模式：直接处理指定 discussion。"""
    process_discussion(
        client,
        qa_engine,
        owner,
        repo,
        discussion_number,
        qa_category_id,
        dry_run=dry_run,
    )


# ── 入口 ─────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSM Q&A Discussion Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--event",
        action="store_true",
        help="从 GITHUB_EVENT_PATH 读取 discussion 事件 payload（Actions event 触发时使用）",
    )
    group.add_argument(
        "--discussion-number",
        type=int,
        metavar="N",
        help="手动指定要处理的 Discussion 编号（workflow_dispatch 或本地调试时使用）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印生成的回答，不实际发布评论",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # ── 读取 Token ──────────────────────────────────────────────────────────
    gh_token = os.environ.get("CSM_QA_API_KEY", "").strip()
    if not gh_token:
        logger.error("CSM_QA_API_KEY 未配置（Fine-grained PAT）")
        return 1

    # ── 初始化 ───────────────────────────────────────────────────────────────
    try:
        owner, repo = _get_repo_parts()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    client = GitHubGraphQL(gh_token)

    try:
        qa_category_id = resolve_qa_category_id(client, owner, repo)
    except RuntimeError as exc:
        logger.error("无法解析 Q&A category: %s", exc)
        return 1

    try:
        qa_engine = CSM_QA.from_env()
    except Exception as exc:
        logger.error("CSM_QA 初始化失败: %s", exc)
        return 1

    # ── 执行 ─────────────────────────────────────────────────────────────────
    try:
        if args.event:
            run_event_mode(
                client, qa_engine, owner, repo, qa_category_id, dry_run=args.dry_run
            )
        else:
            run_manual_mode(
                client,
                qa_engine,
                owner,
                repo,
                qa_category_id,
                args.discussion_number,
                dry_run=args.dry_run,
            )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("未预期的错误: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
