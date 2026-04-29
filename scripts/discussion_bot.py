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
CSM_QA_GH_TOKEN  GitHub Fine-grained PAT（需要 repository discussions: read & write 权限）
LLM_API_KEY      DeepSeek / 其他 LLM 的 API Key
GITHUB_REPOSITORY  格式 owner/repo，如 NEVSTOP-LAB/CSM-QA-Robot（Actions 自动注入）
GITHUB_EVENT_PATH  event 模式时由 Actions 自动注入
DISCUSSION_SOURCE_REPO
    （可选）组织级 Q&A discussions 实际归属的源仓库 owner/repo，
    默认 ``<org>/.github``，其中 ``<org>`` 取自 ``GITHUB_REPOSITORY``。
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

try:
    # 仅用于在扫描模式下识别 LLM 鉴权错误（如 DeepSeek API key 无效），
    # 这是 fatal 错误：所有后续 discussion 的 LLM 调用都会失败，应立即终止扫描。
    from openai import AuthenticationError as _OpenAIAuthError  # type: ignore
except ImportError:  # pragma: no cover - openai 是必装依赖，但保险起见兜底
    _OpenAIAuthError = None  # type: ignore

# ── 确保包根目录在 sys.path（直接运行 scripts/ 时使用）──────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from csm_qa import CSM_QA  # noqa: E402

logger = logging.getLogger("discussion_bot")


def _configure_logging() -> None:
    """配置根日志（仅在脚本直接运行时调用，避免作为库导入时污染全局日志配置）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

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
            raise ValueError("GitHub token (CSM_QA_GH_TOKEN) 未配置")
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


def _get_source_repo_parts() -> tuple[str, str]:
    """解析组织级 Q&A discussions 实际归属的"源仓库"。

    GitHub 上 ``/orgs/<org>/discussions/...`` 这种组织级 discussions 在底层
    其实存储于某个具体仓库（通常是 ``<org>/.github``）。本函数返回该源仓库
    的 ``(owner, repo)``，优先读取环境变量 ``DISCUSSION_SOURCE_REPO``，
    缺省时回退为 ``<org>/.github``，其中 ``<org>`` 取自 ``GITHUB_REPOSITORY``。

    Raises:
        ValueError: 环境变量配置不正确。
    """
    src = os.environ.get("DISCUSSION_SOURCE_REPO", "").strip()
    if src:
        if "/" not in src:
            raise ValueError(
                f"DISCUSSION_SOURCE_REPO 格式不正确: {src!r}，期望 'owner/repo'"
            )
        owner, repo = src.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        if not owner or not repo:
            raise ValueError(
                f"DISCUSSION_SOURCE_REPO 格式不正确: {src!r}，期望 'owner/repo'"
            )
        return owner, repo
    org, _ = _get_repo_parts()
    return org, ".github"


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


def _fetch_more_comments(
    client: GitHubGraphQL, discussion_id: str, after_cursor: str
) -> dict[str, Any]:
    """拉取 Discussion 的后续评论页（用于分页）。

    Returns:
        ``{"nodes": [...], "pageInfo": {"hasNextPage": bool, "endCursor": str|None}}``
    """
    gql = """
    query($discussionId: ID!, $after: String!) {
      node(id: $discussionId) {
        ... on Discussion {
          comments(first: 100, after: $after) {
            nodes {
              id
              body
              createdAt
              author { login }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
      }
    }
    """
    data = client.query(gql, {"discussionId": discussion_id, "after": after_cursor})
    return data.get("node", {}).get(
        "comments",
        {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}},
    )


def fetch_discussion(
    client: GitHubGraphQL, owner: str, repo: str, number: int
) -> dict[str, Any]:
    """拉取指定 discussion 的详情（含所有评论，自动分页）。

    通过 ``pageInfo { hasNextPage endCursor }`` 分页拉取，直至拿到全部评论，
    从而确保 :func:`has_bot_replied` 能正确检测已有 Bot 回复。
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
          closed
          category {
            id
            name
          }
          comments(first: 100) {
            nodes {
              id
              body
              createdAt
              author { login }
            }
            pageInfo {
              hasNextPage
              endCursor
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

    # 分页拉取剩余评论
    while disc["comments"]["pageInfo"]["hasNextPage"]:
        cursor = disc["comments"]["pageInfo"]["endCursor"]
        more = _fetch_more_comments(client, disc["id"], cursor)
        disc["comments"]["nodes"].extend(more.get("nodes", []))
        disc["comments"]["pageInfo"] = more.get(
            "pageInfo", {"hasNextPage": False, "endCursor": None}
        )

    return disc


def get_viewer_login(client: GitHubGraphQL) -> Optional[str]:
    """查询当前 PAT 对应的 GitHub 账号（用于作者身份校验）。

    Returns:
        登录名字符串，或 ``None``（查询失败时）。
    """
    gql = """
    query {
      viewer { login }
    }
    """
    try:
        data = client.query(gql)
        return data.get("viewer", {}).get("login") or None
    except RuntimeError as exc:
        logger.warning("无法获取 viewer 登录名，将跳过作者校验: %s", exc)
        return None


def has_bot_replied(discussion: dict, bot_login: Optional[str] = None) -> bool:
    """检查 Discussion 评论中是否已含 BOT_MARKER（可选：同时校验作者身份）。

    Args:
        discussion: :func:`fetch_discussion` 返回的 discussion dict。
        bot_login: Bot 的 GitHub 登录名；若提供，则只有 **作者为 bot_login**
            且含 BOT_MARKER 的评论才算"Bot 已回复"。不提供时仅按 marker 判断。
    """
    for comment in discussion.get("comments", {}).get("nodes", []):
        body = comment.get("body") or ""
        if BOT_MARKER not in body:
            continue
        if bot_login is None:
            return True
        author = (comment.get("author") or {}).get("login", "")
        if author == bot_login:
            return True
    return False


def _is_bot_comment(comment: dict, bot_login: Optional[str]) -> bool:
    """判断单条评论是否为 Bot 自动生成。

    严格 fail-closed：必须同时满足"含 BOT_MARKER"且"作者 == bot_login"才认定为 Bot。
    若调用方未提供 ``bot_login``（如 viewer 查询失败），则一律返回 ``False``，
    避免普通用户在评论中粘贴 marker 就能伪装为 assistant 注入到对话历史 / 干扰
    follow-up 判定。``has_bot_replied`` 仍保留宽松匹配以防止重复回复。
    """
    body = comment.get("body") or ""
    if BOT_MARKER not in body:
        return False
    if bot_login is None:
        return False
    author = (comment.get("author") or {}).get("login", "")
    return author == bot_login


def compute_reply_plan(
    discussion: dict, bot_login: Optional[str] = None
) -> Optional[tuple[str, list[dict[str, str]]]]:
    """决定是否需要回复，以及回复时的 question / history。

    判定规则（评论默认按创建时间升序返回）：

    * 若 discussion 中尚无 Bot 回复 → 视为新问题，``question = title + body``，``history = []``。
    * 若已有 Bot 回复，且最后一条 Bot 回复之后存在用户的新追问 → 取追问中最后一条作为
      ``question``，并把"原帖 + 中间所有评论（不含本次追问）"按 user/assistant 顺序
      组装为 ``history``。
    * 否则（已回复且无后续追问）→ 返回 ``None`` 表示无需回复。

    Returns:
        ``(question, history)`` 或 ``None``。``history`` 元素形如
        ``{"role": "user"|"assistant", "content": str}``。
    """
    title = (discussion.get("title") or "").strip()
    body = (discussion.get("body") or "").strip()
    original_question = f"{title}\n\n{body}".strip() if body else title

    comments = discussion.get("comments", {}).get("nodes", []) or []

    # 安全保护：若 bot_login 未知（viewer 查询失败），无法可靠区分 Bot 真实回复
    # 与用户伪造的 marker，故只要任意评论含 BOT_MARKER 就跳过整个 discussion，
    # 既避免重复回复（marker 可能确实来自之前真实 Bot），也防止伪造内容被注入
    # 到 follow-up 的 history 中。
    if bot_login is None:
        if any(BOT_MARKER in (c.get("body") or "") for c in comments):
            return None
        return original_question, []

    # 找到最后一条 Bot 回复的索引
    last_bot_idx = -1
    for i, c in enumerate(comments):
        if _is_bot_comment(c, bot_login):
            last_bot_idx = i

    if last_bot_idx == -1:
        # 从未回复过 → 新问题
        return original_question, []

    # 已回复：检查最后一条 Bot 回复之后是否存在用户的追问（任何非 Bot 评论）
    followup_user_indices = [
        i
        for i in range(last_bot_idx + 1, len(comments))
        if not _is_bot_comment(comments[i], bot_login)
    ]
    if not followup_user_indices:
        return None  # 已回复且无追问 → 跳过

    latest_followup_idx = followup_user_indices[-1]
    latest_followup_body = (comments[latest_followup_idx].get("body") or "").strip()
    if not latest_followup_body:
        # 追问内容为空，视为无效
        return None

    # 组装 history：原帖作为首条 user，再依次把 [0, latest_followup_idx) 范围内
    # 的评论按 bot/user 角色串成 assistant/user，最后那条追问留作 question。
    history: list[dict[str, str]] = [
        {"role": "user", "content": original_question}
    ]
    for i in range(latest_followup_idx):
        c_body = (comments[i].get("body") or "").strip()
        if not c_body:
            continue
        role = "assistant" if _is_bot_comment(comments[i], bot_login) else "user"
        history.append({"role": role, "content": c_body})

    return latest_followup_body, history


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


def resolve_org_qa_category_id(
    client: GitHubGraphQL, source_owner: str, source_repo: str
) -> str:
    """通过 GraphQL 查询组织级 Q&A discussions 源仓库中名为 'Q&A' 的
    Discussion Category node ID。

    GitHub 的 "Organization discussions"（``/orgs/<org>/discussions/``）实际
    存储于某个源仓库（通常是 ``<org>/.github``）。这里直接查询源仓库的
    ``discussionCategories``，比扫描 ``Organization.repositoryDiscussions``
    既快又稳，且只需要对源仓库的 Discussions: Read 权限。

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
    data = client.query(gql, {"owner": source_owner, "repo": source_repo})
    repo_data = data.get("repository")
    if repo_data is None:
        raise RuntimeError(
            f"无法访问源仓库 {source_owner}/{source_repo}：仓库不存在或 PAT "
            f"未授权该仓库的 Discussions: Read 权限"
        )
    nodes = (repo_data.get("discussionCategories") or {}).get("nodes", []) or []
    for node in nodes:
        if node.get("name", "").strip() == QA_CATEGORY_NAME:
            logger.info(
                "找到组织 Q&A category（源仓库 %s/%s）: id=%s slug=%s isAnswerable=%s",
                source_owner,
                source_repo,
                node["id"],
                node.get("slug"),
                node.get("isAnswerable"),
            )
            return node["id"]
    available = [n.get("name") for n in nodes]
    raise RuntimeError(
        f"未找到源仓库 {source_owner}/{source_repo} 中名为 {QA_CATEGORY_NAME!r} 的 "
        f"Discussion Category，已有分类: {available}"
    )


def fetch_org_discussion(
    client: GitHubGraphQL, source_owner: str, source_repo: str, number: int
) -> dict[str, Any]:
    """拉取指定组织级 discussion 的详情（含所有评论，自动分页）。

    通过源仓库 ``repository(...).discussion(number:)`` 直接定位，避免扫描
    ``Organization.repositoryDiscussions`` 的高 GraphQL cost。

    Raises:
        RuntimeError: Discussion 不存在或无权限。
    """
    return fetch_discussion(client, source_owner, source_repo, number)


def scan_org_qa_discussions(
    client: GitHubGraphQL,
    source_owner: str,
    source_repo: str,
    category_id: str,
    page_size: int = 50,
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    """全量扫描源仓库中 Q&A 分类下的 Discussion（含所有评论，自动分页）。

    会跳过 ``closed = True`` 的 discussion，仅返回 open 状态的 discussion。
    通过 ``pageInfo`` 翻页，最多扫描 ``max_pages * page_size`` 条 discussion，
    以避免 PAT 配额异常时陷入死循环。

    GitHub GraphQL 的 ``Repository.discussions`` 字段支持 ``categoryId``
    参数，因此服务端就能完成分类过滤，效率显著高于
    ``Organization.repositoryDiscussions`` + 客户端过滤。
    """
    gql = """
    query($owner: String!, $repo: String!, $categoryId: ID!, $pageSize: Int!, $cursor: String) {
      repository(owner: $owner, name: $repo) {
        discussions(
          first: $pageSize,
          after: $cursor,
          categoryId: $categoryId,
          orderBy: {field: CREATED_AT, direction: DESC}
        ) {
          nodes {
            id
            number
            title
            body
            url
            closed
            category {
              id
              name
            }
            comments(first: 100) {
              nodes {
                id
                body
                createdAt
                author { login }
              }
              pageInfo {
                hasNextPage
                endCursor
              }
            }
          }
          pageInfo {
            hasNextPage
            endCursor
          }
        }
      }
    }
    """
    cursor: Optional[str] = None
    all_open: list[dict[str, Any]] = []
    for _ in range(max_pages):
        data = client.query(
            gql,
            {
                "owner": source_owner,
                "repo": source_repo,
                "categoryId": category_id,
                "pageSize": page_size,
                "cursor": cursor,
            },
        )
        repo_data = data.get("repository")
        if repo_data is None:
            raise RuntimeError(
                f"无法访问源仓库 {source_owner}/{source_repo}：仓库不存在或 PAT "
                f"未授权该仓库的 Discussions: Read 权限"
            )
        discussions_data = repo_data.get("discussions") or {}
        nodes = discussions_data.get("nodes", []) or []
        page_info = discussions_data.get(
            "pageInfo", {"hasNextPage": False, "endCursor": None}
        )
        for disc in nodes:
            if disc.get("closed"):
                continue
            # 拉取该 discussion 的全部评论
            while disc["comments"]["pageInfo"]["hasNextPage"]:
                inner_cursor = disc["comments"]["pageInfo"]["endCursor"]
                more = _fetch_more_comments(client, disc["id"], inner_cursor)
                disc["comments"]["nodes"].extend(more.get("nodes", []))
                disc["comments"]["pageInfo"] = more.get(
                    "pageInfo", {"hasNextPage": False, "endCursor": None}
                )
            all_open.append(disc)
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
    return all_open


def _process_discussion_dict(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    discussion: dict[str, Any],
    qa_category_id: str,
    *,
    dry_run: bool = False,
    bot_login: Optional[str] = None,
) -> bool:
    """已取得 discussion dict 后的核心处理逻辑。

    Returns:
        True 表示已回复（或 dry-run 打印），False 表示跳过。
    """
    number = discussion.get("number")
    disc_category_id = discussion.get("category", {}).get("id", "")
    disc_category_name = discussion.get("category", {}).get("name", "")

    if disc_category_id != qa_category_id:
        logger.info(
            "Discussion #%d 分类为 %r，非 Q&A，跳过",
            number,
            disc_category_name,
        )
        return False

    if discussion.get("closed"):
        logger.info("Discussion #%d 已关闭，跳过", number)
        return False

    plan = compute_reply_plan(discussion, bot_login=bot_login)
    if plan is None:
        logger.info("Discussion #%d 已有 Bot 回复且无新追问，跳过", number)
        return False

    question, history = plan
    title = discussion.get("title", "").strip()
    disc_id = discussion["id"]
    disc_url = discussion.get("url", f"#{number}")

    if history:
        logger.info(
            "正在为 Discussion #%d 生成追问回答（history=%d 条）: %r",
            number,
            len(history),
            title,
        )
    else:
        logger.info("正在为 Discussion #%d 生成回答: %r", number, title)

    answer = qa_engine.ask(question, history=history)
    reply_body = build_reply(answer)

    if dry_run:
        logger.info("=== [DRY-RUN] Discussion #%d (%s) ===", number, disc_url)
        logger.info("--- 回复内容 ---\n%s", reply_body)
        return True

    post_comment(client, disc_id, reply_body)
    logger.info("Discussion #%d 已成功回复", number)
    return True


def process_discussion(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    owner: str,
    repo: str,
    number: int,
    qa_category_id: str,
    *,
    dry_run: bool = False,
    bot_login: Optional[str] = None,
) -> bool:
    """处理单条仓库 Discussion，返回 True 表示已回复（或 dry-run 打印），False 表示跳过。"""
    discussion = fetch_discussion(client, owner, repo, number)
    return _process_discussion_dict(
        client, qa_engine, discussion, qa_category_id,
        dry_run=dry_run, bot_login=bot_login,
    )


# ── 触发模式处理 ─────────────────────────────────────────────────────────────


def run_event_mode(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    owner: str,
    repo: str,
    qa_category_id: str,
    *,
    dry_run: bool = False,
    bot_login: Optional[str] = None,
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
        bot_login=bot_login,
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
    bot_login: Optional[str] = None,
) -> None:
    """手动模式：直接处理指定仓库 Discussion。"""
    process_discussion(
        client,
        qa_engine,
        owner,
        repo,
        discussion_number,
        qa_category_id,
        dry_run=dry_run,
        bot_login=bot_login,
    )


def run_org_discussion_mode(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    source_owner: str,
    source_repo: str,
    org_qa_category_id: str,
    discussion_number: int,
    *,
    dry_run: bool = False,
    bot_login: Optional[str] = None,
) -> None:
    """手动模式：直接处理指定组织级 Discussion（位于源仓库 source_owner/source_repo）。"""
    discussion = fetch_org_discussion(
        client, source_owner, source_repo, discussion_number
    )
    _process_discussion_dict(
        client, qa_engine, discussion, org_qa_category_id,
        dry_run=dry_run, bot_login=bot_login,
    )


def run_scan_mode(
    client: GitHubGraphQL,
    qa_engine: CSM_QA,
    source_owner: str,
    source_repo: str,
    org_qa_category_id: str,
    *,
    dry_run: bool = False,
    bot_login: Optional[str] = None,
) -> int:
    """全量扫描模式：处理源仓库 Q&A 分类下所有 open discussion，
    自动回复未答复的 discussion 以及 Bot 回复后又出现新追问的 discussion。

    单条 discussion 处理失败不会中止整个扫描；但若遇到 LLM 鉴权失败
    （如 API key 无效），则立即停止——后续调用必然同样失败，继续只是浪费配额。

    Returns:
        失败的 discussion 数量（0 表示全部成功，调用方据此设置进程退出码）。
    """
    discussions = scan_org_qa_discussions(
        client, source_owner, source_repo, org_qa_category_id
    )
    logger.info("找到 %d 条 open 状态的组织 Q&A 讨论", len(discussions))
    replied = 0
    failed = 0
    auth_failure = False
    for disc in discussions:
        number = disc.get("number")
        try:
            if _process_discussion_dict(
                client, qa_engine, disc, org_qa_category_id,
                dry_run=dry_run, bot_login=bot_login,
            ):
                replied += 1
        except Exception as exc:  # noqa: BLE001 — 扫描需对每条 discussion 隔离失败
            failed += 1
            if _OpenAIAuthError is not None and isinstance(exc, _OpenAIAuthError):
                logger.error(
                    "Discussion #%s 处理失败（LLM 鉴权错误，API key 无效）: %s",
                    number, exc,
                )
                logger.error("LLM 鉴权失败为 fatal，停止扫描剩余讨论")
                auth_failure = True
                break
            logger.exception("Discussion #%s 处理失败，继续处理下一条: %s", number, exc)
    logger.info(
        "本次扫描共回复 %d 条讨论，失败 %d 条%s",
        replied, failed,
        "（因鉴权错误提前终止）" if auth_failure else "",
    )
    return failed


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
        help="手动指定要处理的仓库 Discussion 编号（workflow_dispatch 或本地调试时使用）",
    )
    group.add_argument(
        "--org-discussion-number",
        type=int,
        metavar="N",
        help="手动指定要处理的组织级 Discussion 编号",
    )
    group.add_argument(
        "--scan-org",
        action="store_true",
        help="全量扫描组织 Q&A 分类下所有 open discussion，"
             "回复未答复的以及 Bot 回复后又有新追问的 discussion（手动 workflow_dispatch 使用）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印生成的回答，不实际发布评论",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _configure_logging()
    args = parse_args(argv)

    # ── 读取 Token ──────────────────────────────────────────────────────────
    gh_token = os.environ.get("CSM_QA_GH_TOKEN", "").strip()
    if not gh_token:
        logger.error("CSM_QA_GH_TOKEN 未配置（GitHub Fine-grained PAT）")
        return 1

    # ── 初始化 ───────────────────────────────────────────────────────────────
    try:
        owner, repo = _get_repo_parts()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    client = GitHubGraphQL(gh_token)

    # ── 获取 Bot 登录名（用于作者身份校验，失败时降级为仅 marker 检测）──────
    bot_login = get_viewer_login(client)
    if bot_login:
        logger.info("Bot 账号: %s", bot_login)
    else:
        logger.warning("无法获取 Bot 账号，has_bot_replied 将仅按 marker 检测")

    try:
        qa_engine = CSM_QA.from_env()
    except Exception as exc:
        logger.error("CSM_QA 初始化失败: %s", exc)
        return 1

    # ── 执行 ─────────────────────────────────────────────────────────────────
    try:
        if args.event:
            qa_category_id = resolve_qa_category_id(client, owner, repo)
            run_event_mode(
                client,
                qa_engine,
                owner,
                repo,
                qa_category_id,
                dry_run=args.dry_run,
                bot_login=bot_login,
            )
        elif args.discussion_number is not None:
            qa_category_id = resolve_qa_category_id(client, owner, repo)
            run_manual_mode(
                client,
                qa_engine,
                owner,
                repo,
                qa_category_id,
                args.discussion_number,
                dry_run=args.dry_run,
                bot_login=bot_login,
            )
        elif args.org_discussion_number is not None:
            try:
                source_owner, source_repo = _get_source_repo_parts()
            except ValueError as exc:
                logger.error("%s", exc)
                return 1
            org_category_id = resolve_org_qa_category_id(
                client, source_owner, source_repo
            )
            run_org_discussion_mode(
                client,
                qa_engine,
                source_owner,
                source_repo,
                org_category_id,
                args.org_discussion_number,
                dry_run=args.dry_run,
                bot_login=bot_login,
            )
        else:
            # --scan-org
            try:
                source_owner, source_repo = _get_source_repo_parts()
            except ValueError as exc:
                logger.error("%s", exc)
                return 1
            org_category_id = resolve_org_qa_category_id(
                client, source_owner, source_repo
            )
            scan_failures = run_scan_mode(
                client,
                qa_engine,
                source_owner,
                source_repo,
                org_category_id,
                dry_run=args.dry_run,
                bot_login=bot_login,
            )
            if scan_failures:
                # 至少一条 discussion 处理失败：以非零码退出，让 Actions 标记失败，
                # 同时保留前面已成功回复的进度（不抛异常以避免再触发 traceback）。
                return 1
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("未预期的错误: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
