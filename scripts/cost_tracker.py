# -*- coding: utf-8 -*-
"""
费用监控模块 — Token 计数与预算日报
=====================================

实施计划关联：AI-012 费用监控 — Token 计数与预算日报
参考文档：docs/调研/07-费用评估.md

功能：
- 记录每次 LLM 调用的 token 消耗和费用
- 每日费用累计与预算告警
- 月度费用汇总

数据格式：
    cost_log.jsonl 每行一条记录：
    {"timestamp": "...", "model": "...", "prompt_tokens": N,
     "completion_tokens": N, "cache_hit_tokens": N, "usd_cost": 0.001}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CostTracker:
    """费用追踪器

    实施计划关联：AI-012

    记录 LLM 调用费用到 jsonl 文件，支持当日累计和月度汇总。

    Args:
        data_dir: 数据目录路径
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cost_log_path = self.data_dir / "cost_log.jsonl"
        self.cost_summary_path = self.data_dir / "cost_summary.json"

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: int,
        usd_cost: float,
    ):
        """记录一次 LLM 调用的费用

        追加到 cost_log.jsonl 文件。

        Args:
            model: 使用的模型名称
            prompt_tokens: 输入 token 数
            completion_tokens: 输出 token 数
            cache_hit_tokens: 缓存命中 token 数
            usd_cost: 本次调用费用（USD）
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "date": date.today().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "usd_cost": round(usd_cost, 8),
        }

        try:
            with open(self.cost_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"写入费用日志失败: {e}")

    def get_daily_cost(self, target_date: date | None = None) -> float:
        """获取指定日期的累计费用

        Args:
            target_date: 目标日期（默认今天）

        Returns:
            当日累计费用（USD）
        """
        target = (target_date or date.today()).isoformat()
        total = 0.0

        if not self.cost_log_path.exists():
            return 0.0

        try:
            with open(self.cost_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("date") == target:
                        total += entry.get("usd_cost", 0)
        except Exception as e:
            logger.error(f"读取费用日志失败: {e}")

        return total

    def get_daily_summary(self, target_date: date | None = None) -> dict:
        """获取指定日期的费用摘要

        Args:
            target_date: 目标日期（默认今天）

        Returns:
            包含 total_cost, call_count, total_prompt_tokens 等的字典
        """
        target = (target_date or date.today()).isoformat()
        summary = {
            "date": target,
            "total_cost_usd": 0.0,
            "call_count": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cache_hit_tokens": 0,
        }

        if not self.cost_log_path.exists():
            return summary

        try:
            with open(self.cost_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("date") == target:
                        summary["total_cost_usd"] += entry.get("usd_cost", 0)
                        summary["call_count"] += 1
                        summary["total_prompt_tokens"] += entry.get(
                            "prompt_tokens", 0
                        )
                        summary["total_completion_tokens"] += entry.get(
                            "completion_tokens", 0
                        )
                        summary["total_cache_hit_tokens"] += entry.get(
                            "cache_hit_tokens", 0
                        )
        except Exception as e:
            logger.error(f"读取费用日志失败: {e}")

        return summary

    def update_monthly_summary(self):
        """更新月度费用汇总

        按月累计写入 data/cost_summary.json

        Returns:
            月度汇总字典
        """
        monthly: dict[str, float] = {}

        if self.cost_log_path.exists():
            try:
                with open(self.cost_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        d = entry.get("date", "")
                        if len(d) >= 7:
                            month = d[:7]  # YYYY-MM
                            monthly[month] = monthly.get(month, 0) + entry.get(
                                "usd_cost", 0
                            )
            except Exception as e:
                logger.error(f"读取费用日志失败: {e}")

        try:
            with open(self.cost_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"monthly_cost_usd": monthly},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"写入月度汇总失败: {e}")

        return monthly

    def print_daily_report(self, target_date: date | None = None):
        """输出当日费用报告到 stdout

        实施计划关联：AI-012 任务 2
        """
        summary = self.get_daily_summary(target_date)
        print(f"\n=== 费用日报 ({summary['date']}) ===")
        print(f"  API 调用次数: {summary['call_count']}")
        print(f"  总费用: ${summary['total_cost_usd']:.6f}")
        print(f"  输入 tokens: {summary['total_prompt_tokens']}")
        print(f"  输出 tokens: {summary['total_completion_tokens']}")
        print(f"  缓存命中 tokens: {summary['total_cache_hit_tokens']}")
        cache_rate = 0
        if summary["total_prompt_tokens"] > 0:
            cache_rate = (
                summary["total_cache_hit_tokens"]
                / summary["total_prompt_tokens"]
                * 100
            )
        print(f"  缓存命中率: {cache_rate:.1f}%")
        print("=" * 35)
