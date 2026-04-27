"""
归档管理模块
参考: docs/plan/README.md § AI-009, docs/调研/05-回复归档与存储.md

功能：
- list_pending(): 列出 pending/ 目录下待审核的回复文件
- approve_pending(): 审核通过并归档
- reject_pending(): 拒绝并归档
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def approve_pending(filepath: Path) -> bool:
    """
    审核通过：移动到已完成目录

    Args:
        filepath: pending 文件路径

    Returns:
        True 表示处理成功
    """
    try:
        # 移动到已完成目录（data/done/，与线程归档目录分离）
        # 从 pending 文件所在路径推算 data/ 根目录：
        # pending 文件在 data/pending/ 目录下
        done_dir = filepath.parent.parent / "done"
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
