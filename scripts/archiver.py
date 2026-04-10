"""
归档管理模块
参考: docs/plan/README.md § AI-009, docs/调研/05-回复归档与存储.md

功能：
- list_pending(): 列出 pending/ 目录下待审核的回复文件
- approve_pending(): 审核通过并发布
- reject_pending(): 拒绝并归档
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import frontmatter

logger = logging.getLogger(__name__)


def _map_object_type(object_type: str) -> str:
    """将配置中的 object_type 映射为知乎 API 接受的类型

    ZhihuClient.post_comment() 约定使用 article/answer；
    配置中的 question 在发布前映射为 answer。
    """
    if object_type == "question":
        return "answer"
    return object_type


def list_pending(pending_dir: str) -> list[Path]:
    """
    列出 pending/ 目录下所有待审核文件

    Args:
        pending_dir: pending 目录路径

    Returns:
        待审核文件路径列表
    """
    path = Path(pending_dir)
    if not path.exists():
        return []

    files = sorted(path.glob("*.md"))
    logger.info("待审核回复: %d 个", len(files))
    return files


def approve_pending(filepath: Path, zhihu_client=None) -> bool:
    """
    审核通过：发布评论并移动到已处理目录

    Args:
        filepath: pending 文件路径
        zhihu_client: 知乎客户端（None 时跳过发布）

    Returns:
        True 表示处理成功
    """
    try:
        post = frontmatter.load(str(filepath))
        meta = post.metadata

        if zhihu_client and meta.get("status") == "pending":
            # 解析回复内容（"## 生成的回复" 之后的内容）
            content = post.content
            reply_start = content.find("## 生成的回复")
            if reply_start >= 0:
                reply_text = content[reply_start + len("## 生成的回复"):].strip()
            else:
                reply_text = content.strip()

            success = zhihu_client.post_comment(
                meta.get("article_id", ""),
                # 从 pending 元数据中读取 object_type；
                # question 在发布时映射为 answer
                _map_object_type(meta.get("object_type", "article")),
                reply_text,
                parent_id=meta.get("comment_id"),
            )

            if not success:
                logger.warning("发布失败: %s", filepath.name)
                return False

        # 移动到已完成目录
        done_dir = filepath.parent.parent / "archive" / "done"
        done_dir.mkdir(parents=True, exist_ok=True)
        filepath.rename(done_dir / filepath.name)

        logger.info("审核通过: %s", filepath.name)
        return True

    except Exception as e:
        logger.error("审核处理异常: %s - %s", filepath.name, e)
        return False


def reject_pending(filepath: Path, reason: str = "") -> bool:
    """
    拒绝回复：移动到已拒绝目录

    Args:
        filepath: pending 文件路径
        reason: 拒绝原因

    Returns:
        True 表示处理成功
    """
    try:
        rejected_dir = filepath.parent.parent / "archive" / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        filepath.rename(rejected_dir / filepath.name)

        logger.info("回复已拒绝: %s (原因: %s)", filepath.name, reason)
        return True

    except Exception as e:
        logger.error("拒绝处理异常: %s - %s", filepath.name, e)
        return False
