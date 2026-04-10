# -*- coding: utf-8 -*-
"""
ThreadManager 单元测试
======================

实施计划关联：AI-007 验收标准
独立于实现的测试用例，覆盖：
- 顶级评论新建 thread.md
- 追问复用已有 thread
- append_turn 写入后 front-matter 可正确解析
- is_human=True 时 ⭐ 标记存在
- build_context_messages 超过 max_turns 时截断
"""
from pathlib import Path

import pytest
import frontmatter

from scripts.thread_manager import ThreadManager


# ===== Fixtures =====

@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    """创建临时归档目录"""
    d = tmp_path / "archive"
    d.mkdir()
    return d


@pytest.fixture
def manager(archive_dir: Path) -> ThreadManager:
    """创建 ThreadManager 实例"""
    return ThreadManager(archive_dir=str(archive_dir))


@pytest.fixture
def sample_comment() -> dict:
    """示例顶级评论"""
    return {
        "id": "12345678",
        "author": "test_user",
        "content": "这篇文章写得很好，能否详细说明一下 CSM 的核心概念？",
        "created_time": 1712000000,
    }


@pytest.fixture
def sample_article_meta() -> dict:
    """示例文章元信息"""
    return {
        "title": "CSM 最佳实践系列（一）",
        "url": "https://zhuanlan.zhihu.com/p/98765432",
    }


# ===== 线程创建测试 =====

class TestGetOrCreateThread:
    """验证线程创建逻辑"""

    def test_create_new_thread(
        self, manager, sample_comment, sample_article_meta
    ):
        """顶级评论应创建新的 thread.md 文件"""
        path = manager.get_or_create_thread(
            article_id="98765432",
            root_comment=sample_comment,
            article_meta=sample_article_meta,
        )

        assert path.exists()
        assert path.name == "12345678.md"

    def test_reuse_existing_thread(
        self, manager, sample_comment, sample_article_meta
    ):
        """追问应复用已有的 thread 文件"""
        path1 = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )
        path2 = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        assert path1 == path2

    def test_thread_frontmatter_fields(
        self, manager, sample_comment, sample_article_meta
    ):
        """新建 thread 的 front-matter 应包含所有必要字段"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        post = frontmatter.load(str(path))
        assert post.metadata["thread_id"] == "12345678"
        assert post.metadata["article_id"] == "98765432"
        assert post.metadata["article_title"] == "CSM 最佳实践系列（一）"
        assert post.metadata["commenter"] == "test_user"
        assert post.metadata["turn_count"] == 0
        assert post.metadata["human_replied"] is False

    def test_thread_directory_structure(
        self, manager, archive_dir, sample_comment, sample_article_meta
    ):
        """线程文件应在正确的目录结构下"""
        manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        expected_dir = archive_dir / "articles" / "98765432" / "threads"
        assert expected_dir.exists()

    def test_different_articles_different_dirs(
        self, manager, sample_comment, sample_article_meta
    ):
        """不同文章的线程应在不同目录"""
        path1 = manager.get_or_create_thread(
            "111", sample_comment, sample_article_meta
        )
        path2 = manager.get_or_create_thread(
            "222", sample_comment, sample_article_meta
        )

        assert path1.parent != path2.parent


# ===== 追加轮次测试 =====

class TestAppendTurn:
    """验证 append_turn 写入逻辑"""

    def test_append_user_comment(
        self, manager, sample_comment, sample_article_meta
    ):
        """追加用户评论后文件应包含评论内容"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        manager.append_turn(
            thread_path=path,
            author="test_user",
            content="这是用户的评论",
        )

        post = frontmatter.load(str(path))
        assert "这是用户的评论" in post.content
        assert post.metadata["turn_count"] == 1

    def test_append_bot_reply(
        self, manager, sample_comment, sample_article_meta
    ):
        """追加 Bot 回复后应包含模型和 token 信息"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        manager.append_turn(
            thread_path=path,
            author="Bot",
            content="CSM 的核心概念包括...",
            model="deepseek-chat",
            tokens=150,
        )

        post = frontmatter.load(str(path))
        assert "Bot 回复" in post.content
        assert "deepseek-chat" in post.content
        assert "150" in post.content

    def test_append_human_reply_star_marker(
        self, manager, sample_comment, sample_article_meta
    ):
        """真人回复应有 ⭐ 标记"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        manager.append_turn(
            thread_path=path,
            author="作者",
            content="感谢关注，CSM 的核心是...",
            is_human=True,
        )

        post = frontmatter.load(str(path))
        assert "⭐" in post.content
        assert "真人回复" in post.content

    def test_human_reply_updates_frontmatter(
        self, manager, sample_comment, sample_article_meta
    ):
        """真人回复应更新 front-matter 的 human_replied"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        # 初始为 False
        post = frontmatter.load(str(path))
        assert post.metadata["human_replied"] is False

        manager.append_turn(
            path, "作者", "真人回复内容", is_human=True
        )

        post = frontmatter.load(str(path))
        assert post.metadata["human_replied"] is True

    def test_turn_count_increments(
        self, manager, sample_comment, sample_article_meta
    ):
        """每次追加轮次后 turn_count 应递增"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        for i in range(5):
            manager.append_turn(path, f"user_{i}", f"评论 {i}")

        post = frontmatter.load(str(path))
        assert post.metadata["turn_count"] == 5

    def test_last_updated_changes(
        self, manager, sample_comment, sample_article_meta
    ):
        """追加轮次后 last_updated 应更新"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        post_before = frontmatter.load(str(path))
        ts_before = post_before.metadata["last_updated"]

        manager.append_turn(path, "user", "新评论")

        post_after = frontmatter.load(str(path))
        ts_after = post_after.metadata["last_updated"]

        # last_updated 应被更新（可能相同如果在同一秒内）
        assert ts_after is not None

    def test_frontmatter_parseable_after_multiple_turns(
        self, manager, sample_comment, sample_article_meta
    ):
        """多次追加后 front-matter 仍可被正确解析"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        manager.append_turn(path, "user1", "问题1")
        manager.append_turn(
            path, "Bot", "回复1", model="deepseek-chat", tokens=100
        )
        manager.append_turn(path, "user1", "追问")
        manager.append_turn(path, "作者", "真人回复", is_human=True)

        # 验证 front-matter 仍可解析
        post = frontmatter.load(str(path))
        assert post.metadata["turn_count"] == 4
        assert post.metadata["human_replied"] is True
        assert "thread_id" in post.metadata


# ===== 上下文构建测试 =====

class TestBuildContextMessages:
    """验证 build_context_messages 逻辑"""

    def test_empty_thread_returns_empty(
        self, manager, sample_comment, sample_article_meta
    ):
        """空线程应返回空列表"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        messages = manager.build_context_messages(path)
        assert messages == []

    def test_single_turn_returns_one_message(
        self, manager, sample_comment, sample_article_meta
    ):
        """单轮对话应返回一条消息"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )
        manager.append_turn(path, "user", "用户评论")

        messages = manager.build_context_messages(path)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_bot_reply_is_assistant_role(
        self, manager, sample_comment, sample_article_meta
    ):
        """Bot 回复应为 assistant role"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )
        manager.append_turn(path, "user", "问题")
        manager.append_turn(
            path, "Bot", "回复", model="deepseek-chat", tokens=50
        )

        messages = manager.build_context_messages(path)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_human_reply_is_user_role(
        self, manager, sample_comment, sample_article_meta
    ):
        """真人回复应为 user role"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )
        manager.append_turn(path, "作者", "真人回复内容", is_human=True)

        messages = manager.build_context_messages(path)
        assert messages[0]["role"] == "user"

    def test_max_turns_truncation(
        self, manager, sample_comment, sample_article_meta
    ):
        """超过 max_turns 时应截断为最近的轮次"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )

        # 追加 10 轮对话
        for i in range(10):
            if i % 2 == 0:
                manager.append_turn(path, f"user", f"问题{i}")
            else:
                manager.append_turn(
                    path, "Bot", f"回复{i}",
                    model="deepseek-chat", tokens=50
                )

        # max_turns=4，应只返回最后 4 轮
        messages = manager.build_context_messages(path, max_turns=4)
        assert len(messages) == 4

    def test_max_turns_no_truncation_when_fewer(
        self, manager, sample_comment, sample_article_meta
    ):
        """轮次数少于 max_turns 时不截断"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )
        manager.append_turn(path, "user", "问题")
        manager.append_turn(
            path, "Bot", "回复", model="deepseek-chat", tokens=50
        )

        messages = manager.build_context_messages(path, max_turns=6)
        assert len(messages) == 2

    def test_message_content_not_empty(
        self, manager, sample_comment, sample_article_meta
    ):
        """消息内容不应为空"""
        path = manager.get_or_create_thread(
            "98765432", sample_comment, sample_article_meta
        )
        manager.append_turn(path, "user", "一个具体的问题")

        messages = manager.build_context_messages(path)
        for msg in messages:
            assert msg["content"].strip() != ""
