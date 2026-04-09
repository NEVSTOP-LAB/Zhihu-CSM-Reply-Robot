"""
AI-008: 评论前置过滤器
参考: docs/plan/README.md § AI-008

功能：
- should_skip(): 判断评论是否应跳过（广告、重复、超长等）
- truncate_comment(): 截断超长评论

过滤规则（按优先级）：
1. 广告/敏感词：命中 spam_keywords 时跳过
2. 重复评论：同一 author 在 dedup_window_minutes 内重复出现时跳过
3. 超长截断：评论 token 数 > max_comment_tokens 时截断（不跳过）

设计说明：
- 使用 tiktoken 精确计算 token 数
- 重复检测基于内存中的作者+时间窗口缓存
- 截断后的评论仍然可以被处理（返回 skip=False）
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)

# 全局重复检测缓存：{author: last_comment_timestamp}
_dedup_cache: dict[str, float] = {}


def should_skip(comment: object, settings: dict) -> tuple[bool, str]:
    """
    判断评论是否需要跳过
    参考: docs/plan/README.md § AI-008 第 1 点

    Args:
        comment: 评论对象（需有 author, content, created_time 属性或键）
        settings: 配置字典（含 filter 配置段）

    Returns:
        (是否跳过, 原因) 元组
        跳过时返回 (True, "原因说明")
        不跳过时返回 (False, "")
    """
    filter_settings = settings.get("filter", {})

    # 获取评论属性（支持 dict 和 dataclass）
    content = _get_attr(comment, "content", "")
    author = _get_attr(comment, "author", "")
    created_time = _get_attr(comment, "created_time", 0)

    # 规则 1: 广告/敏感词检测
    # 参考: docs/plan/README.md § AI-008 第 2 点 — 广告关键词
    spam_keywords = filter_settings.get("spam_keywords", [])
    for keyword in spam_keywords:
        if keyword.lower() in content.lower():
            logger.info("评论被过滤（广告关键词 '%s'）: author=%s", keyword, author)
            return True, f"广告关键词: {keyword}"

    # 规则 2: 重复评论检测
    # 参考: docs/plan/README.md § AI-008 第 2 点 — 同一 author 在 dedup_window_minutes 内重复
    dedup_minutes = filter_settings.get("dedup_window_minutes", 60)
    dedup_seconds = dedup_minutes * 60
    current_time = created_time or time.time()

    if author in _dedup_cache:
        last_time = _dedup_cache[author]
        if current_time - last_time < dedup_seconds:
            logger.info(
                "评论被过滤（重复评论，%d分钟内）: author=%s",
                dedup_minutes, author,
            )
            return True, f"重复评论: {author} 在 {dedup_minutes} 分钟内已有评论"

    # 更新重复检测缓存
    _dedup_cache[author] = current_time

    return False, ""


def truncate_comment(content: str, max_tokens: int = 500) -> tuple[str, bool]:
    """
    截断超长评论
    参考: docs/plan/README.md § AI-008 第 2 点 — 超长截断
    参考: docs/调研/06-Token优化策略.md § 6. 截断超长评论

    使用 tiktoken 精确计算 token 数（离线不可用时降级为字符估算）。

    Args:
        content: 评论内容
        max_tokens: 最大 token 数（默认 500）

    Returns:
        (截断后的内容, 是否被截断) 元组
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(content)
        if len(tokens) <= max_tokens:
            return content, False
        truncated = encoding.decode(tokens[:max_tokens])
        logger.info("评论被截断: %d → %d tokens", len(tokens), max_tokens)
        return truncated + "...", True
    except Exception:
        # 离线环境降级：按字符估算（中文约 1.5 token/字）
        max_chars = int(max_tokens * 0.67)  # 保守估算
        if len(content) <= max_chars:
            return content, False
        logger.info("评论被截断（字符估算）: %d → %d 字符", len(content), max_chars)
        return content[:max_chars] + "...", True


def reset_dedup_cache() -> None:
    """重置重复检测缓存（用于测试或新一天开始时）"""
    global _dedup_cache
    _dedup_cache = {}


def _get_attr(obj: object, name: str, default=None):
    """从对象或字典中获取属性"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
