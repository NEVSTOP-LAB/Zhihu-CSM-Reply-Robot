"""
AI-008: CommentFilter 单元测试
参考: docs/plan/README.md § AI-008 测试要求

测试覆盖：
- 超长评论被截断（不跳过）
- 广告关键词命中时跳过
- 60分钟内同一用户第二条评论被跳过
- 正常评论不被过滤
"""

import time

import pytest

from scripts.comment_filter import should_skip, truncate_comment, reset_dedup_cache


# ─── fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_dedup_cache():
    """每个测试前清空重复检测缓存"""
    reset_dedup_cache()
    yield
    reset_dedup_cache()


@pytest.fixture
def default_settings() -> dict:
    """默认过滤配置"""
    return {
        "filter": {
            "max_comment_tokens": 500,
            "spam_keywords": ["加微信", "私信", "VX", "wx"],
            "dedup_window_minutes": 60,
        }
    }


def make_comment(
    content: str = "正常评论内容",
    author: str = "测试用户",
    created_time: float = 0,
) -> dict:
    """创建测试评论"""
    return {
        "content": content,
        "author": author,
        "created_time": created_time or time.time(),
    }


# ─── 广告关键词测试 ──────────────────────────────────────────────

class TestSpamFilter:
    """测试广告关键词过滤"""

    def test_spam_keyword_skip(self, default_settings) -> None:
        """包含广告关键词的评论应被跳过"""
        comment = make_comment(content="加微信详聊")
        skip, reason = should_skip(comment, default_settings)
        assert skip is True
        assert "广告关键词" in reason

    def test_spam_keyword_case_insensitive(self, default_settings) -> None:
        """广告关键词匹配应不区分大小写"""
        comment = make_comment(content="可以加WX吗")
        skip, reason = should_skip(comment, default_settings)
        assert skip is True

    def test_multiple_spam_keywords(self, default_settings) -> None:
        """每种广告关键词都应被检测"""
        for keyword in ["加微信", "私信我", "加VX", "加wx"]:
            reset_dedup_cache()
            comment = make_comment(content=keyword, author=f"user_{keyword}")
            skip, _ = should_skip(comment, default_settings)
            assert skip is True, f"关键词 '{keyword}' 未被检测"

    def test_normal_content_passes(self, default_settings) -> None:
        """正常内容不应被过滤"""
        comment = make_comment(content="请问 CSM 如何处理客户投诉？")
        skip, reason = should_skip(comment, default_settings)
        assert skip is False
        assert reason == ""


# ─── 重复评论检测测试 ──────────────────────────────────────────

class TestDedupFilter:
    """测试重复评论检测"""

    def test_first_comment_passes(self, default_settings) -> None:
        """用户的第一条评论不应被过滤"""
        comment = make_comment(author="新用户", created_time=time.time())
        skip, _ = should_skip(comment, default_settings)
        assert skip is False

    def test_second_comment_within_window_skipped(self, default_settings) -> None:
        """60分钟内同一用户第二条评论应被跳过"""
        now = time.time()
        comment1 = make_comment(author="重复用户", created_time=now)
        comment2 = make_comment(author="重复用户", created_time=now + 30 * 60)  # 30分钟后

        should_skip(comment1, default_settings)  # 第一条通过
        skip, reason = should_skip(comment2, default_settings)
        assert skip is True
        assert "重复评论" in reason

    def test_comment_after_window_passes(self, default_settings) -> None:
        """超过 dedup_window 后的评论不应被过滤"""
        now = time.time()
        comment1 = make_comment(author="间隔用户", created_time=now)
        comment2 = make_comment(
            author="间隔用户",
            created_time=now + 61 * 60,  # 61分钟后
        )

        should_skip(comment1, default_settings)
        skip, _ = should_skip(comment2, default_settings)
        assert skip is False

    def test_different_users_independent(self, default_settings) -> None:
        """不同用户的评论应独立检测"""
        now = time.time()
        comment_a = make_comment(author="用户A", created_time=now)
        comment_b = make_comment(author="用户B", created_time=now + 5)

        should_skip(comment_a, default_settings)
        skip, _ = should_skip(comment_b, default_settings)
        assert skip is False


# ─── 超长截断测试 ──────────────────────────────────────────────

class TestTruncation:
    """测试超长评论截断"""

    def test_short_comment_not_truncated(self) -> None:
        """短评论不应被截断"""
        content = "简短的评论"
        result, truncated = truncate_comment(content, max_tokens=500)
        assert result == content
        assert truncated is False

    def test_long_comment_truncated(self) -> None:
        """超长评论应被截断"""
        # 生成一段很长的文本
        content = "这是一段很长的评论。" * 200  # 远超 500 tokens
        result, truncated = truncate_comment(content, max_tokens=50)
        assert truncated is True
        assert result.endswith("...")
        assert len(result) < len(content)

    def test_truncation_preserves_meaning(self) -> None:
        """截断应保留前部分内容"""
        content = "第一部分内容。" * 100
        result, truncated = truncate_comment(content, max_tokens=20)
        assert truncated is True
        assert result.startswith("第一部分内容")

    def test_truncation_does_not_skip(self, default_settings) -> None:
        """超长评论截断后不应被跳过（should_skip 返回 False）"""
        comment = make_comment(content="正常内容" * 200)
        skip, _ = should_skip(comment, default_settings)
        # should_skip 只做跳过检测，不做截断
        # 截断由主流程调用 truncate_comment 单独处理
        assert skip is False


# ─── 综合测试 ──────────────────────────────────────────────────

class TestIntegration:
    """综合测试"""

    def test_normal_comment_full_pass(self, default_settings) -> None:
        """正常评论应完全通过所有过滤规则"""
        comment = make_comment(
            content="关于 CSM 的问题，请问客户流失率如何计算？",
            author="技术用户",
            created_time=time.time(),
        )
        skip, reason = should_skip(comment, default_settings)
        assert skip is False
        assert reason == ""

    def test_spam_priority_over_dedup(self, default_settings) -> None:
        """广告检测优先于重复检测"""
        now = time.time()
        comment = make_comment(
            content="加微信详聊",
            author="广告用户",
            created_time=now,
        )
        skip, reason = should_skip(comment, default_settings)
        assert skip is True
        assert "广告关键词" in reason  # 应该是广告原因而不是重复

    def test_empty_settings_no_filter(self) -> None:
        """空配置时不应过滤任何评论"""
        comment = make_comment(content="任何内容")
        skip, _ = should_skip(comment, {})
        assert skip is False
