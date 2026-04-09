"""
AI-007: ThreadManager 单元测试
参考: docs/plan/README.md § AI-007 测试要求

测试覆盖：
- 顶级评论新建 thread.md，追问复用已有 thread
- append_turn 写入后 front-matter 可被 python-frontmatter 正确解析
- is_human=True 时 ⭐ 标记存在
- build_context_messages 超过 max_turns 时截断
"""

from pathlib import Path

import frontmatter
import pytest

from scripts.thread_manager import ThreadManager


# ─── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    """创建临时归档目录"""
    d = tmp_path / "archive"
    d.mkdir()
    return d


@pytest.fixture
def manager(archive_dir: Path) -> ThreadManager:
    """创建测试用 ThreadManager"""
    return ThreadManager(archive_dir=str(archive_dir))


@pytest.fixture
def sample_comment() -> dict:
    """样本评论"""
    return {
        "id": "12345678",
        "author": "测试用户",
        "content": "请问 CSM 如何处理客户投诉？",
        "created_time": 1700000000,
    }


@pytest.fixture
def sample_article_meta() -> dict:
    """样本文章元数据"""
    return {
        "title": "CSM 最佳实践系列（一）",
        "url": "https://zhuanlan.zhihu.com/p/98765432",
    }


# ─── get_or_create_thread 测试 ──────────────────────────────────

class TestGetOrCreateThread:
    """测试线程文件创建"""

    def test_create_new_thread(self, manager, sample_comment, sample_article_meta) -> None:
        """顶级评论应创建新的 thread.md 文件"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)

        assert path.exists()
        assert path.name == "12345678.md"
        assert "articles/98765432/threads" in str(path)

    def test_reuse_existing_thread(self, manager, sample_comment, sample_article_meta) -> None:
        """追问应复用已有 thread（同一 thread_id）"""
        path1 = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        path2 = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)

        assert path1 == path2

    def test_thread_frontmatter_parseable(self, manager, sample_comment, sample_article_meta) -> None:
        """创建的线程文件 front-matter 应可被 python-frontmatter 正确解析"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        post = frontmatter.load(str(path))

        assert post.metadata["thread_id"] == "12345678"
        assert post.metadata["article_id"] == "98765432"
        assert post.metadata["article_title"] == "CSM 最佳实践系列（一）"
        assert post.metadata["commenter"] == "测试用户"
        assert post.metadata["turn_count"] == 0
        assert post.metadata["human_replied"] is False

    def test_different_comments_different_threads(self, manager, sample_article_meta) -> None:
        """不同顶级评论应创建不同的线程文件"""
        comment1 = {"id": "111", "author": "用户A", "content": "问题1", "created_time": 100}
        comment2 = {"id": "222", "author": "用户B", "content": "问题2", "created_time": 200}

        path1 = manager.get_or_create_thread("98765432", comment1, sample_article_meta)
        path2 = manager.get_or_create_thread("98765432", comment2, sample_article_meta)

        assert path1 != path2
        assert path1.name == "111.md"
        assert path2.name == "222.md"


# ─── append_turn 测试 ──────────────────────────────────────────

class TestAppendTurn:
    """测试对话追加"""

    def test_append_user_comment(self, manager, sample_comment, sample_article_meta) -> None:
        """追加用户评论后文件内容应包含评论"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(
            path, author="测试用户", content="请问如何处理投诉？",
            comment_id="12345678",
        )

        post = frontmatter.load(str(path))
        assert post.metadata["turn_count"] == 1
        assert "请问如何处理投诉？" in post.content

    def test_append_bot_reply(self, manager, sample_comment, sample_article_meta) -> None:
        """追加机器人回复后应包含模型信息"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(
            path, author="机器人", content="处理投诉的步骤是...",
            model="deepseek-chat", tokens=150,
        )

        content = path.read_text(encoding="utf-8")
        assert "机器人回复" in content
        assert "deepseek-chat" in content
        assert "处理投诉的步骤是..." in content

    def test_append_human_reply_star_marker(self, manager, sample_comment, sample_article_meta) -> None:
        """is_human=True 时应添加 ⭐ 标记"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(
            path, author="作者", content="真人的详细回复",
            is_human=True,
        )

        content = path.read_text(encoding="utf-8")
        assert "⭐" in content
        assert "真人回复" in content

    def test_human_reply_updates_frontmatter(self, manager, sample_comment, sample_article_meta) -> None:
        """is_human=True 时应更新 front-matter human_replied=true"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(
            path, author="作者", content="回复",
            is_human=True,
        )

        post = frontmatter.load(str(path))
        assert post.metadata["human_replied"] is True

    def test_frontmatter_parseable_after_append(self, manager, sample_comment, sample_article_meta) -> None:
        """多次追加后 front-matter 仍可正确解析"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)

        manager.append_turn(path, "用户", "第一条", comment_id="1")
        manager.append_turn(path, "机器人", "回复1", model="deepseek-chat", tokens=100)
        manager.append_turn(path, "用户", "追问", is_followup=True, comment_id="2")
        manager.append_turn(path, "作者", "真人回复", is_human=True)

        post = frontmatter.load(str(path))
        assert post.metadata["turn_count"] == 4
        assert post.metadata["human_replied"] is True

    def test_turn_count_increments(self, manager, sample_comment, sample_article_meta) -> None:
        """每次追加 turn_count 应递增"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)

        for i in range(5):
            manager.append_turn(path, f"用户{i}", f"内容{i}")

        post = frontmatter.load(str(path))
        assert post.metadata["turn_count"] == 5


# ─── build_context_messages 测试 ──────────────────────────────

class TestBuildContextMessages:
    """测试对话上下文构建"""

    def test_build_messages_format(self, manager, sample_comment, sample_article_meta) -> None:
        """构建的 messages 应为 OpenAI 格式"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(path, "用户", "问题内容", comment_id="1")
        manager.append_turn(path, "机器人", "回复内容", model="deepseek-chat")

        messages = manager.build_context_messages(path)
        assert len(messages) >= 1
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("user", "assistant")

    def test_user_message_role(self, manager, sample_comment, sample_article_meta) -> None:
        """用户评论应映射为 user role"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(path, "用户", "问一个问题", comment_id="1")

        messages = manager.build_context_messages(path)
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_bot_message_role(self, manager, sample_comment, sample_article_meta) -> None:
        """机器人/真人回复应映射为 assistant role"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(path, "用户", "问题", comment_id="1")
        manager.append_turn(path, "机器人", "回复", model="deepseek-chat")

        messages = manager.build_context_messages(path)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1

    def test_max_turns_truncation(self, manager, sample_comment, sample_article_meta) -> None:
        """超过 max_turns 时应截断，只保留最近的轮次"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)

        # 添加 10 轮对话
        for i in range(10):
            manager.append_turn(path, f"用户{i}", f"问题{i}", comment_id=str(i))
            manager.append_turn(path, "机器人", f"回复{i}", model="model")

        # 限制 4 轮
        messages = manager.build_context_messages(path, max_turns=4)
        assert len(messages) <= 4

    def test_empty_thread_returns_empty(self, manager, sample_comment, sample_article_meta) -> None:
        """空线程应返回空列表"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        messages = manager.build_context_messages(path)
        assert messages == []

    def test_human_reply_as_assistant(self, manager, sample_comment, sample_article_meta) -> None:
        """真人回复也应映射为 assistant role"""
        path = manager.get_or_create_thread("98765432", sample_comment, sample_article_meta)
        manager.append_turn(path, "用户", "问题", comment_id="1")
        manager.append_turn(path, "作者", "真人回复内容", is_human=True)

        messages = manager.build_context_messages(path)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
