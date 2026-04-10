# -*- coding: utf-8 -*-
"""
评论前置过滤器
==============

实施计划关联：AI-008 前置过滤器 — 边界情况处理
参考文档：docs/Review-方案/ 评审建议

功能：
- 超长评论截断（不跳过，截断后继续处理）
- 广告/敏感词过滤（跳过）
- 重复评论检测（同一用户在时间窗口内重复，跳过）
- 自动跳过感谢类评论

使用方式：
    filter = CommentFilter(settings)
    should_skip, reason = filter.should_skip(comment)
    if should_skip:
        log(f"跳过评论: {reason}")
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


class CommentFilter:
    """评论前置过滤器

    实施计划关联：AI-008

    在主流程处理评论前进行过滤，支持：
    - 超长截断（>max_comment_tokens 时截断，不跳过）
    - 广告关键词（命中 spam_keywords 时跳过）
    - 重复评论（同一 author 在 dedup_window_minutes 内跳过）
    - 感谢类评论（命中 auto_skip_patterns 时跳过）

    Args:
        settings: 配置字典（来自 config/settings.yaml）
    """

    def __init__(self, settings: dict):
        filter_cfg = settings.get("filter", {})
        review_cfg = settings.get("review", {})

        self.max_comment_tokens = filter_cfg.get("max_comment_tokens", 500)
        self.spam_keywords = filter_cfg.get("spam_keywords", [])
        self.dedup_window_minutes = filter_cfg.get("dedup_window_minutes", 60)
        self.auto_skip_patterns = [
            re.compile(p) for p in review_cfg.get("auto_skip_patterns", [])
        ]

        # 最近评论记录：{author: last_timestamp}
        # 用于重复评论检测
        self._recent_comments: dict[str, float] = {}

        # Token 编码器（用于超长评论检测）
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def _count_tokens(self, text: str) -> int:
        """计算文本 token 数

        使用 tiktoken cl100k_base 编码器。
        如果编码器不可用，使用粗略的字符数估算。
        """
        if self._encoder:
            return len(self._encoder.encode(text))
        # 粗略估算：中文约 1 字 ≈ 1.5 tokens
        return int(len(text) * 1.5)

    def truncate_if_needed(self, content: str) -> str:
        """如果评论超长则截断

        实施计划关联：AI-008 任务 2（超长截断阈值）

        超过 max_comment_tokens 的评论会被截断，
        但不会被跳过（截断后继续处理）。

        Args:
            content: 评论内容

        Returns:
            原内容或截断后的内容
        """
        token_count = self._count_tokens(content)
        if token_count <= self.max_comment_tokens:
            return content

        # 按 token 截断
        if self._encoder:
            tokens = self._encoder.encode(content)
            truncated_tokens = tokens[:self.max_comment_tokens]
            truncated = self._encoder.decode(truncated_tokens)
        else:
            # 粗略截断
            max_chars = int(self.max_comment_tokens / 1.5)
            truncated = content[:max_chars]

        logger.info(
            f"评论超长截断: {token_count} → {self.max_comment_tokens} tokens"
        )
        return truncated + "..."

    def should_skip(
        self, comment: dict, current_time: float | None = None
    ) -> tuple[bool, str]:
        """判断是否应跳过该评论

        实施计划关联：AI-008 任务 1

        按以下顺序检查过滤规则：
        1. 广告/敏感词 → 跳过
        2. 感谢类评论 → 跳过
        3. 重复评论 → 跳过
        4. 超长 → 不跳过（截断由 truncate_if_needed 处理）

        Args:
            comment: 评论字典 {content, author, created_time}
            current_time: 当前时间戳（测试用，默认 time.time()）

        Returns:
            (是否跳过, 原因说明)
        """
        content = comment.get("content", "")
        author = comment.get("author", "")
        timestamp = (
            current_time
            if current_time is not None
            else comment.get("created_time", time.time())
        )

        # 1. 广告/敏感词检查
        for keyword in self.spam_keywords:
            if keyword in content:
                logger.info(f"跳过广告评论: keyword={keyword}, author={author}")
                return True, f"广告关键词: {keyword}"

        # 2. 感谢类评论检查
        for pattern in self.auto_skip_patterns:
            if pattern.match(content.strip()):
                logger.info(f"跳过感谢评论: author={author}")
                return True, "感谢类评论"

        # 3. 重复评论检查
        if author in self._recent_comments:
            last_time = self._recent_comments[author]
            elapsed_minutes = (timestamp - last_time) / 60
            if elapsed_minutes < self.dedup_window_minutes:
                logger.info(
                    f"跳过重复评论: author={author}, "
                    f"距上次 {elapsed_minutes:.0f} 分钟"
                )
                return True, f"重复评论（{elapsed_minutes:.0f}分钟内）"

        # 记录本次评论时间
        self._recent_comments[author] = timestamp

        # 4. 通过所有过滤
        return False, ""
