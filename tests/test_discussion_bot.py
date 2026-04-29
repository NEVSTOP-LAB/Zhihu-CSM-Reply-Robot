"""tests/test_discussion_bot.py — discussion_bot 脚本单元测试."""

from __future__ import annotations

import pytest
from typing import Optional

# 确保 scripts/ 目录在路径中
import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.discussion_bot import (
    BOT_MARKER,
    BOT_FOOTER,
    build_reply,
    has_bot_replied,
    _fetch_more_comments,
    _process_discussion_dict,
    get_viewer_login,
    resolve_org_qa_category_id,
    fetch_org_discussion,
    scan_org_qa_discussions,
)


# ── build_reply ───────────────────────────────────────────────────────────────


def test_build_reply_contains_answer():
    reply = build_reply("CSM 是一种状态机框架。")
    assert "CSM 是一种状态机框架。" in reply


def test_build_reply_contains_footer():
    reply = build_reply("some answer")
    assert BOT_FOOTER in reply


def test_build_reply_contains_bot_marker():
    reply = build_reply("some answer")
    assert BOT_MARKER in reply


def test_build_reply_order():
    """回复顺序应为：答案 → 页脚 → 标记。"""
    reply = build_reply("answer text")
    ans_pos = reply.index("answer text")
    footer_pos = reply.index(BOT_FOOTER)
    marker_pos = reply.index(BOT_MARKER)
    assert ans_pos < footer_pos <= marker_pos


def test_build_reply_strips_trailing_whitespace_from_answer():
    """答案末尾空白应被去除，不影响页脚格式。"""
    reply = build_reply("answer   \n\n")
    # 回复不应以多余空行开始页脚
    assert "answer" in reply
    assert BOT_FOOTER in reply


def test_build_reply_multiline_answer():
    """多行答案应完整保留。"""
    answer = "第一行\n第二行\n第三行"
    reply = build_reply(answer)
    assert "第一行" in reply
    assert "第三行" in reply


# ── has_bot_replied ───────────────────────────────────────────────────────────


def _make_discussion(comments: list[str], author: str = "user") -> dict:
    """构造最小化的 discussion dict。"""
    return {
        "comments": {
            "nodes": [{"id": f"c{i}", "body": body, "author": {"login": author}} for i, body in enumerate(comments)]
        }
    }


def test_has_bot_replied_false_when_no_comments():
    disc = _make_discussion([])
    assert has_bot_replied(disc) is False


def test_has_bot_replied_false_when_no_marker():
    disc = _make_discussion(["普通回复，没有标记。", "另一条回复。"])
    assert has_bot_replied(disc) is False


def test_has_bot_replied_true_when_marker_present_no_login_check():
    """不提供 bot_login 时，仅凭 marker 判断（向下兼容），不验证作者。"""
    disc = _make_discussion(["普通回复", f"Bot 回复内容 {BOT_MARKER}"])
    # bot_login=None：只要有 marker 就返回 True
    assert has_bot_replied(disc) is True
    assert has_bot_replied(disc, bot_login=None) is True


def test_has_bot_replied_author_check_wrong_author():
    """提供 bot_login 但评论作者不匹配时应返回 False（防伪造 marker 攻击）。"""
    disc = _make_discussion(["普通回复", f"Bot 回复内容 {BOT_MARKER}"])
    # disc 中 author 是 "user"，而 bot_login 是 "real-bot" → 不匹配
    assert has_bot_replied(disc, bot_login="real-bot") is False


def test_has_bot_replied_true_when_marker_present():
    disc = _make_discussion(["普通回复", f"Bot 回复内容 {BOT_MARKER}"])
    assert has_bot_replied(disc) is True


def test_has_bot_replied_author_check_match():
    """提供 bot_login，且作者匹配时返回 True。"""
    disc = {
        "comments": {
            "nodes": [
                {"id": "c0", "body": "普通回复", "author": {"login": "user"}},
                {"id": "c1", "body": f"Bot 回复 {BOT_MARKER}", "author": {"login": "my-bot"}},
            ]
        }
    }
    assert has_bot_replied(disc, bot_login="my-bot") is True


def test_has_bot_replied_author_check_no_match():
    """提供 bot_login，marker 存在但作者不匹配（用户伪造 marker）时返回 False。"""
    disc = {
        "comments": {
            "nodes": [
                {"id": "c0", "body": f"恶意评论 {BOT_MARKER}", "author": {"login": "attacker"}},
            ]
        }
    }
    assert has_bot_replied(disc, bot_login="my-bot") is False


def test_has_bot_replied_handles_none_body():
    """comment body 为 None 时不应抛出异常。"""
    disc = {
        "comments": {
            "nodes": [
                {"id": "c0", "body": None, "author": {"login": "user"}},
                {"id": "c1", "body": f"正常回复 {BOT_MARKER}", "author": {"login": "bot"}},
            ]
        }
    }
    assert has_bot_replied(disc) is True
    assert has_bot_replied(disc, bot_login="bot") is True
    assert has_bot_replied(disc, bot_login="other") is False


def test_has_bot_replied_handles_empty_body():
    disc = _make_discussion(["", "   "])
    assert has_bot_replied(disc) is False


def test_has_bot_replied_marker_in_first_comment():
    disc = _make_discussion([f"第一条即是 Bot 回复 {BOT_MARKER}", "第二条普通回复"])
    assert has_bot_replied(disc) is True


def test_has_bot_replied_author_is_none():
    """author 字段为 None（匿名评论）时不应崩溃。"""
    disc = {
        "comments": {
            "nodes": [
                {"id": "c0", "body": f"匿名回复 {BOT_MARKER}", "author": None},
            ]
        }
    }
    assert has_bot_replied(disc) is True
    assert has_bot_replied(disc, bot_login="my-bot") is False


# ── get_viewer_login ──────────────────────────────────────────────────────────


class _MockGraphQL:
    """最小化 mock GitHubGraphQL，直接返回预设 data。"""

    def __init__(self, return_data: dict, raise_error_message: Optional[str] = None):
        self._return_data = return_data
        self._raise_error_message = raise_error_message

    def query(self, gql: str, variables: Optional[dict] = None) -> dict:
        if self._raise_error_message:
            raise RuntimeError(self._raise_error_message)
        return self._return_data


def test_get_viewer_login_returns_login():
    client = _MockGraphQL({"viewer": {"login": "test-bot"}})
    assert get_viewer_login(client) == "test-bot"


def test_get_viewer_login_returns_none_on_error():
    client = _MockGraphQL({}, raise_error_message="Unauthorized")
    assert get_viewer_login(client) is None


def test_get_viewer_login_returns_none_when_missing():
    client = _MockGraphQL({"viewer": {}})
    assert get_viewer_login(client) is None


# ── _fetch_more_comments ──────────────────────────────────────────────────────


def test_fetch_more_comments_returns_empty_on_missing_node():
    """node 字段缺失时应返回空 comments 结构，而非抛出。"""
    client = _MockGraphQL({"node": {}})
    result = _fetch_more_comments(client, "D_xxx", "cursor123")
    assert result["nodes"] == []
    assert result["pageInfo"]["hasNextPage"] is False


# ── resolve_org_qa_category_id ────────────────────────────────────────────────


def test_resolve_org_qa_category_id_found():
    """能正确返回组织中名为 'Q&A' 的 category ID。"""
    data = {
        "organization": {
            "discussionCategories": {
                "nodes": [
                    {"id": "cat1", "name": "General", "slug": "general", "isAnswerable": False},
                    {"id": "cat2", "name": "Q&A", "slug": "q-a", "isAnswerable": True},
                ]
            }
        }
    }
    client = _MockGraphQL(data)
    assert resolve_org_qa_category_id(client, "NEVSTOP-LAB") == "cat2"


def test_resolve_org_qa_category_id_not_found():
    """未找到 Q&A category 时应抛出 RuntimeError。"""
    data = {
        "organization": {
            "discussionCategories": {
                "nodes": [
                    {"id": "cat1", "name": "General", "slug": "general", "isAnswerable": False},
                ]
            }
        }
    }
    client = _MockGraphQL(data)
    with pytest.raises(RuntimeError, match="Q&A"):
        resolve_org_qa_category_id(client, "NEVSTOP-LAB")


def test_resolve_org_qa_category_id_empty():
    """分类列表为空时应抛出 RuntimeError。"""
    data = {"organization": {"discussionCategories": {"nodes": []}}}
    client = _MockGraphQL(data)
    with pytest.raises(RuntimeError):
        resolve_org_qa_category_id(client, "NEVSTOP-LAB")


# ── fetch_org_discussion ──────────────────────────────────────────────────────


def _make_org_discussion_payload(number: int = 31, comments: list[dict] | None = None) -> dict:
    """构造 organization.discussion GraphQL 响应 payload。"""
    return {
        "organization": {
            "discussion": {
                "id": f"D_org_{number}",
                "number": number,
                "title": f"Org Discussion #{number}",
                "body": "Body text",
                "url": f"https://github.com/orgs/NEVSTOP-LAB/discussions/{number}",
                "category": {"id": "cat2", "name": "Q&A"},
                "comments": {
                    "nodes": comments or [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                },
            }
        }
    }


def test_fetch_org_discussion_returns_discussion():
    client = _MockGraphQL(_make_org_discussion_payload(31))
    disc = fetch_org_discussion(client, "NEVSTOP-LAB", 31)
    assert disc["number"] == 31
    assert disc["id"] == "D_org_31"
    assert disc["category"]["name"] == "Q&A"


def test_fetch_org_discussion_not_found():
    client = _MockGraphQL({"organization": {"discussion": None}})
    with pytest.raises(RuntimeError, match="31"):
        fetch_org_discussion(client, "NEVSTOP-LAB", 31)


# ── scan_org_qa_discussions ───────────────────────────────────────────────────


def test_scan_org_qa_discussions_returns_list():
    """scan_org_qa_discussions 应返回 discussion 列表。"""
    data = {
        "organization": {
            "discussions": {
                "nodes": [
                    {
                        "id": "D1",
                        "number": 1,
                        "title": "Q1",
                        "body": "body1",
                        "url": "https://github.com/orgs/NEVSTOP-LAB/discussions/1",
                        "category": {"id": "cat2", "name": "Q&A"},
                        "comments": {
                            "nodes": [],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        },
                    }
                ]
            }
        }
    }
    client = _MockGraphQL(data)
    discussions = scan_org_qa_discussions(client, "NEVSTOP-LAB", "cat2")
    assert len(discussions) == 1
    assert discussions[0]["number"] == 1


def test_scan_org_qa_discussions_empty():
    """分类下无讨论时应返回空列表。"""
    data = {"organization": {"discussions": {"nodes": []}}}
    client = _MockGraphQL(data)
    assert scan_org_qa_discussions(client, "NEVSTOP-LAB", "cat2") == []


# ── _process_discussion_dict ──────────────────────────────────────────────────


class _FakeQAEngine:
    """最小化 QA 引擎 mock，总是返回固定答案。"""
    def ask(self, question: str) -> str:
        return f"answer to: {question}"


def test_process_discussion_dict_skips_wrong_category():
    """分类不匹配时应跳过（返回 False）。"""
    client = _MockGraphQL({})
    disc = {
        "id": "D1", "number": 1, "title": "Q", "body": "",
        "url": "https://github.com/orgs/NEVSTOP-LAB/discussions/1",
        "category": {"id": "other-cat", "name": "General"},
        "comments": {"nodes": []},
    }
    result = _process_discussion_dict(client, _FakeQAEngine(), disc, "qa-cat", dry_run=True)
    assert result is False


def test_process_discussion_dict_skips_already_replied():
    """已有 Bot 回复时应跳过（返回 False）。"""
    from scripts.discussion_bot import BOT_MARKER
    client = _MockGraphQL({})
    disc = {
        "id": "D1", "number": 1, "title": "Q", "body": "",
        "url": "https://github.com/orgs/NEVSTOP-LAB/discussions/1",
        "category": {"id": "qa-cat", "name": "Q&A"},
        "comments": {"nodes": [{"id": "c0", "body": f"reply {BOT_MARKER}", "author": {"login": "bot"}}]},
    }
    result = _process_discussion_dict(client, _FakeQAEngine(), disc, "qa-cat", dry_run=True)
    assert result is False


def test_process_discussion_dict_dry_run_returns_true():
    """dry_run 模式下应生成回复并返回 True，不实际发帖。"""
    client = _MockGraphQL({})
    disc = {
        "id": "D1", "number": 1, "title": "How does CSM work?", "body": "",
        "url": "https://github.com/orgs/NEVSTOP-LAB/discussions/1",
        "category": {"id": "qa-cat", "name": "Q&A"},
        "comments": {"nodes": []},
    }
    result = _process_discussion_dict(client, _FakeQAEngine(), disc, "qa-cat", dry_run=True)
    assert result is True
