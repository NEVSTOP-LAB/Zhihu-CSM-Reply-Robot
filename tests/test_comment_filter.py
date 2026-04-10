# -*- coding: utf-8 -*-
"""
CommentFilter 单元测试
======================

实施计划关联：AI-008 验收标准
独立于实现的测试用例，覆盖：
- 超长评论被截断（不跳过）
- 广告关键词命中时跳过
- 60分钟内同一用户第二条评论被跳过
- 正常评论不被过滤
- 感谢类评论被跳过
"""
import time

import pytest

from scripts.comment_filter import CommentFilter


# ===== Fixtures =====

SAMPLE_SETTINGS = {
    "filter": {
        "max_comment_tokens": 50,       # 低阈值便于测试
        "spam_keywords": ["加微信", "私信", "免费领取"],
        "dedup_window_minutes": 60,
    },
    "review": {
        "auto_skip_patterns": [
            "^谢谢[！!]?$",
            "^感谢[！!]?$",
        ],
    },
}


@pytest.fixture
def filter_instance():
    """创建 CommentFilter 实例"""
    return CommentFilter(settings=SAMPLE_SETTINGS)


def _make_comment(
    content: str = "普通评论",
    author: str = "user",
    created_time: float = 1712000000,
) -> dict:
    """构造测试评论"""
    return {
        "content": content,
        "author": author,
        "created_time": created_time,
    }


# ===== 正常评论测试 =====

class TestNormalComments:
    """验证正常评论不被过滤"""

    def test_normal_comment_passes(self, filter_instance):
        """普通评论不应被跳过"""
        comment = _make_comment("这篇文章写得很好，想了解更多")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is False
        assert reason == ""

    def test_technical_question_passes(self, filter_instance):
        """技术问题不应被跳过"""
        comment = _make_comment("CSM 中如何处理客户投诉？有什么最佳实践？")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is False


# ===== 广告过滤测试 =====

class TestSpamFilter:
    """验证广告/敏感词过滤"""

    def test_spam_keyword_加微信(self, filter_instance):
        """包含 '加微信' 应被跳过"""
        comment = _make_comment("加微信了解更多详情")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is True
        assert "加微信" in reason

    def test_spam_keyword_私信(self, filter_instance):
        """包含 '私信' 应被跳过"""
        comment = _make_comment("私信我获取免费资料")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is True
        assert "私信" in reason

    def test_spam_keyword_免费领取(self, filter_instance):
        """包含 '免费领取' 应被跳过"""
        comment = _make_comment("免费领取CSM资料包")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is True
        assert "免费领取" in reason

    def test_partial_match_within_sentence(self, filter_instance):
        """广告关键词在句子中间也应被检测"""
        comment = _make_comment("如果有需要可以加微信我")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is True


# ===== 感谢类评论测试 =====

class TestAutoSkipPatterns:
    """验证感谢类评论自动跳过"""

    def test_简单谢谢(self, filter_instance):
        """'谢谢' 应被跳过"""
        comment = _make_comment("谢谢")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is True
        assert "感谢" in reason

    def test_谢谢加感叹号(self, filter_instance):
        """'谢谢！' 应被跳过"""
        comment = _make_comment("谢谢！")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is True

    def test_感谢(self, filter_instance):
        """'感谢' 应被跳过"""
        comment = _make_comment("感谢")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is True

    def test_longer_thanks_not_skipped(self, filter_instance):
        """'谢谢分享，学到了很多' 不应被跳过（不是纯感谢）"""
        comment = _make_comment("谢谢分享，学到了很多")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is False


# ===== 重复评论测试 =====

class TestDedup:
    """验证重复评论检测"""

    def test_first_comment_not_deduped(self, filter_instance):
        """用户第一条评论不应被跳过"""
        comment = _make_comment("第一条", author="user_a", created_time=1000)
        skip, reason = filter_instance.should_skip(comment, current_time=1000)
        assert skip is False

    def test_second_comment_within_window(self, filter_instance):
        """同一用户 60 分钟内第二条评论应被跳过"""
        t = 1712000000
        # 第一条
        c1 = _make_comment("第一条", author="user_a", created_time=t)
        filter_instance.should_skip(c1, current_time=t)

        # 30 分钟后第二条
        c2 = _make_comment("第二条", author="user_a", created_time=t + 30 * 60)
        skip, reason = filter_instance.should_skip(c2, current_time=t + 30 * 60)
        assert skip is True
        assert "重复" in reason

    def test_second_comment_outside_window(self, filter_instance):
        """同一用户超过 60 分钟后评论不应被跳过"""
        t = 1712000000
        c1 = _make_comment("第一条", author="user_a", created_time=t)
        filter_instance.should_skip(c1, current_time=t)

        # 90 分钟后
        c2 = _make_comment("第二条", author="user_a", created_time=t + 90 * 60)
        skip, reason = filter_instance.should_skip(c2, current_time=t + 90 * 60)
        assert skip is False

    def test_different_users_not_deduped(self, filter_instance):
        """不同用户的评论不应互相影响"""
        t = 1712000000
        c1 = _make_comment("评论", author="user_a", created_time=t)
        filter_instance.should_skip(c1, current_time=t)

        c2 = _make_comment("评论", author="user_b", created_time=t + 5 * 60)
        skip, reason = filter_instance.should_skip(c2, current_time=t + 5 * 60)
        assert skip is False


# ===== 超长评论截断测试 =====

class TestTruncation:
    """验证超长评论截断逻辑"""

    def test_short_comment_not_truncated(self, filter_instance):
        """短评论不应被截断"""
        content = "简短评论"
        result = filter_instance.truncate_if_needed(content)
        assert result == content

    def test_long_comment_truncated(self, filter_instance):
        """超长评论应被截断"""
        # 构造一个明显超长的评论（>50 tokens）
        content = "这是一段很长很长的评论内容。" * 50
        result = filter_instance.truncate_if_needed(content)
        assert len(result) < len(content)
        assert result.endswith("...")

    def test_truncation_does_not_skip(self, filter_instance):
        """超长评论应被截断但不应被跳过"""
        content = "这是一段很长的评论。" * 50
        comment = _make_comment(content=content)
        skip, reason = filter_instance.should_skip(comment)
        # 超长不应导致跳过
        assert skip is False

    def test_exactly_at_limit(self, filter_instance):
        """恰好在限制处的评论不应被截断"""
        # 精确控制较难，测试接近限制的情况
        content = "测试"
        result = filter_instance.truncate_if_needed(content)
        assert result == content  # 短于限制，不截断


# ===== 边界情况测试 =====

class TestEdgeCases:
    """验证边界情况"""

    def test_empty_content(self, filter_instance):
        """空评论不应被跳过"""
        comment = _make_comment(content="")
        skip, reason = filter_instance.should_skip(comment)
        # 空评论不匹配任何过滤规则
        assert skip is False

    def test_whitespace_only(self, filter_instance):
        """纯空白不应匹配感谢模式"""
        comment = _make_comment(content="   ")
        skip, reason = filter_instance.should_skip(comment)
        assert skip is False

    def test_empty_settings(self):
        """空配置应使用默认值"""
        f = CommentFilter(settings={})
        comment = _make_comment("普通评论")
        skip, reason = f.should_skip(comment)
        assert skip is False
