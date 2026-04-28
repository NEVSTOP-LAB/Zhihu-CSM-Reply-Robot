"""tests/test_discussion_bot.py — discussion_bot 脚本单元测试."""

from __future__ import annotations

import pytest

# 确保 scripts/ 目录在路径中
import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.discussion_bot import BOT_MARKER, BOT_FOOTER, build_reply, has_bot_replied


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


def _make_discussion(comments: list[str]) -> dict:
    """构造最小化的 discussion dict。"""
    return {
        "comments": {
            "nodes": [{"id": f"c{i}", "body": body, "author": {"login": "user"}} for i, body in enumerate(comments)]
        }
    }


def test_has_bot_replied_false_when_no_comments():
    disc = _make_discussion([])
    assert has_bot_replied(disc) is False


def test_has_bot_replied_false_when_no_marker():
    disc = _make_discussion(["普通回复，没有标记。", "另一条回复。"])
    assert has_bot_replied(disc) is False


def test_has_bot_replied_true_when_marker_present():
    disc = _make_discussion(["普通回复", f"Bot 回复内容 {BOT_MARKER}"])
    assert has_bot_replied(disc) is True


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


def test_has_bot_replied_handles_empty_body():
    disc = _make_discussion(["", "   "])
    assert has_bot_replied(disc) is False


def test_has_bot_replied_marker_in_first_comment():
    disc = _make_discussion([f"第一条即是 Bot 回复 {BOT_MARKER}", "第二条普通回复"])
    assert has_bot_replied(disc) is True
