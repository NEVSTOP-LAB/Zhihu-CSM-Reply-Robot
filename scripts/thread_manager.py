"""
AI-007: ThreadManager — 对话线程文件管理
参考: docs/plan/README.md § AI-007, docs/调研/05-回复归档与存储.md

功能：
1. get_or_create_thread(): 创建或获取对话线程文件
2. append_turn(): 追加一轮对话（机器人/真人），is_human=True 时加 ⭐
3. build_context_messages(): 构建 OpenAI messages 格式的上下文

线程文件格式：
- YAML front-matter（thread_id, article_id, commenter 等元数据）
- Markdown 正文（对话记录，按时间追加）
- 真人回复标记 ⭐，便于快速辨认和索引

设计说明：
- 线程 ID = 顶级评论的 comment_id
- 同一 thread 的追问归入同一文件
- 使用 python-frontmatter 读写 YAML front-matter
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    对话线程文件管理器
    参考: docs/plan/README.md § AI-007, docs/调研/05-回复归档与存储.md

    管理 archive/articles/{article_id}/threads/{thread_id}.md 文件
    """

    def __init__(self, archive_dir: str) -> None:
        """
        初始化线程管理器

        Args:
            archive_dir: 归档根目录路径（如 "archive"）
        """
        self.archive_dir = Path(archive_dir)
        logger.info("ThreadManager 初始化: archive_dir=%s", self.archive_dir)

    def get_or_create_thread(
        self,
        article_id: str,
        root_comment: dict,
        article_meta: dict,
    ) -> Path:
        """
        获取或创建对话线程文件
        参考: docs/plan/README.md § AI-007 第 1 点
        参考: docs/调研/05-回复归档与存储.md § 2. 目录结构设计

        Args:
            article_id: 文章 ID
            root_comment: 顶级评论信息 {"id", "author", "content", "created_time"}
            article_meta: 文章元数据 {"title", "url"}

        Returns:
            线程文件路径
        """
        thread_id = str(root_comment["id"])
        thread_dir = self.archive_dir / "articles" / article_id / "threads"
        thread_path = thread_dir / f"{thread_id}.md"

        if thread_path.exists():
            logger.debug("线程已存在: %s", thread_path)
            return thread_path

        # 创建新线程文件
        thread_dir.mkdir(parents=True, exist_ok=True)

        # 构建 front-matter 元数据
        # 参考: docs/调研/05-回复归档与存储.md § 3. 对话线程文件格式
        now = datetime.now(timezone.utc).isoformat()
        post = frontmatter.Post("")
        post.metadata = {
            "thread_id": thread_id,
            "article_id": article_id,
            "article_title": article_meta.get("title", ""),
            "article_url": article_meta.get("url", ""),
            "article_summary": article_meta.get("summary", ""),
            "commenter": root_comment.get("author", "未知用户"),
            "started_at": now,
            "last_updated": now,
            "turn_count": 0,
            "human_replied": False,
        }
        post.content = "\n## 对话记录\n"

        # 写入文件
        with open(thread_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        logger.info("创建新线程: %s (article=%s)", thread_id, article_id)
        return thread_path

    def append_turn(
        self,
        thread_path: Path,
        author: str,
        content: str,
        is_human: bool = False,
        comment_id: Optional[str] = None,
        model: Optional[str] = None,
        tokens: Optional[int] = None,
        is_followup: bool = False,
    ) -> None:
        """
        追加一轮对话到线程文件
        参考: docs/plan/README.md § AI-007 第 2 点
        参考: docs/调研/05-回复归档与存储.md § 3. 对话线程文件格式

        Args:
            thread_path: 线程文件路径
            author: 作者名称
            content: 对话内容
            is_human: 是否为真人回复（标记 ⭐）
            comment_id: 评论 ID
            model: LLM 模型名（机器人回复时）
            tokens: 使用的 token 数（机器人回复时）
            is_followup: 是否为追问
        """
        # 读取现有文件
        post = frontmatter.load(str(thread_path))

        # 更新元数据
        now = datetime.now(timezone.utc).isoformat()
        post.metadata["last_updated"] = now
        post.metadata["turn_count"] = post.metadata.get("turn_count", 0) + 1

        if is_human:
            # 参考: docs/plan/README.md § AI-007 第 2 点 — is_human=True 时加 ⭐
            post.metadata["human_replied"] = True

        # 格式化时间
        time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        # 构建对话条目
        # 参考: docs/调研/05-回复归档与存储.md § 3. 对话线程文件格式
        if is_human:
            # 真人回复 ⭐
            header = f"\n\n### {time_str} · 真人回复（作者本人）⭐\n"
            body = f"\n{content}\n"
        elif model:
            # 机器人回复
            token_info = f", tokens: {tokens}" if tokens else ""
            header = f"\n\n### {time_str} · Bot 回复（model: {model}{token_info}）\n"
            body = f"\n{content}\n"
        elif is_followup:
            # 用户追问
            cid = f" #{comment_id}" if comment_id else ""
            header = f"\n\n### {time_str} · {author}（追问{cid}）\n"
            body = f"\n> {content}\n"
        else:
            # 用户首评
            cid = f" #{comment_id}" if comment_id else ""
            header = f"\n\n### {time_str} · {author}（评论{cid}）\n"
            body = f"\n> {content}\n"

        # 追加到内容
        post.content += header + body + "\n---\n"

        # 写回文件
        with open(thread_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        logger.info(
            "追加对话: thread=%s, author=%s, is_human=%s",
            thread_path.stem, author, is_human,
        )

    def build_context_messages(
        self,
        thread_path: Path,
        max_turns: int = 6,
    ) -> list[dict]:
        """
        将线程历史转换为 OpenAI messages 格式
        参考: docs/plan/README.md § AI-007 第 3 点
        参考: docs/调研/05-回复归档与存储.md § 4. 追问上下文处理

        最多保留最近 max_turns 轮对话（约 1000~1500 tokens）。

        Args:
            thread_path: 线程文件路径
            max_turns: 最多保留轮数

        Returns:
            OpenAI messages 格式的对话列表
        """
        post = frontmatter.load(str(thread_path))
        content = post.content

        # 解析对话轮次
        # 格式：### 时间 · 角色（类型）\n\n内容
        turns = self._parse_turns(content)

        # 截断到最近 max_turns 轮
        if len(turns) > max_turns:
            turns = turns[-max_turns:]

        # 转换为 OpenAI messages 格式
        messages = []
        for turn in turns:
            role = "assistant" if turn["is_bot"] else "user"
            messages.append({"role": role, "content": turn["content"]})

        logger.debug(
            "构建上下文: thread=%s, turns=%d/%d",
            thread_path.stem, len(messages), post.metadata.get("turn_count", 0),
        )
        return messages

    @staticmethod
    def _parse_turns(content: str) -> list[dict]:
        """
        解析线程内容中的对话轮次

        Args:
            content: 线程文件正文内容

        Returns:
            对话轮次列表，每项含 is_bot 和 content
        """
        turns = []
        # 匹配 ### 开头的对话条目
        sections = re.split(r'\n(?=### )', content)

        for section in sections:
            section = section.strip()
            if not section.startswith("###"):
                continue

            # 判断是 bot（assistant）还是 user（含真人回复和用户评论）
            is_bot = "Bot 回复" in section or "机器人回复" in section

            # 提取内容（去掉标题行）
            lines = section.split("\n", 1)
            body = lines[1].strip() if len(lines) > 1 else ""
            # 去掉 markdown 引用符和分隔线
            body = body.replace("---", "").strip()
            body = re.sub(r'^>\s*', '', body, flags=re.MULTILINE).strip()

            if body:
                turns.append({
                    "is_bot": is_bot,
                    "content": body,
                })

        return turns
