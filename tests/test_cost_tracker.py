# -*- coding: utf-8 -*-
"""
CostTracker 单元测试
====================

实施计划关联：AI-012 验收标准
独立于实现的测试用例，覆盖：
- 多次调用后 cost_log.jsonl 行数正确
- 日费用累计计算正确
- 月度汇总正确
"""
import json
from datetime import date
from pathlib import Path

import pytest

from scripts.cost_tracker import CostTracker


# ===== Fixtures =====

@pytest.fixture
def tracker(tmp_path: Path) -> CostTracker:
    """创建测试用 CostTracker"""
    return CostTracker(data_dir=str(tmp_path / "data"))


# ===== 记录测试 =====

class TestRecord:
    """验证费用记录逻辑"""

    def test_record_creates_file(self, tracker):
        """首次记录应创建 jsonl 文件"""
        tracker.record("deepseek-chat", 100, 50, 0, 0.001)
        assert tracker.cost_log_path.exists()

    def test_record_appends_lines(self, tracker):
        """多次记录应追加行"""
        tracker.record("deepseek-chat", 100, 50, 0, 0.001)
        tracker.record("deepseek-chat", 200, 100, 50, 0.002)
        tracker.record("deepseek-chat", 150, 75, 30, 0.0015)

        lines = tracker.cost_log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_record_valid_json(self, tracker):
        """每行应为有效 JSON"""
        tracker.record("deepseek-chat", 100, 50, 20, 0.001)

        line = tracker.cost_log_path.read_text().strip()
        entry = json.loads(line)
        assert entry["model"] == "deepseek-chat"
        assert entry["prompt_tokens"] == 100
        assert entry["completion_tokens"] == 50
        assert entry["cache_hit_tokens"] == 20
        assert entry["usd_cost"] == 0.001

    def test_record_includes_date(self, tracker):
        """记录应包含日期"""
        tracker.record("deepseek-chat", 100, 50, 0, 0.001)

        line = tracker.cost_log_path.read_text().strip()
        entry = json.loads(line)
        assert "date" in entry
        assert "timestamp" in entry


# ===== 日费用测试 =====

class TestDailyCost:
    """验证日费用累计"""

    def test_daily_cost_empty(self, tracker):
        """无记录时费用为 0"""
        assert tracker.get_daily_cost() == 0.0

    def test_daily_cost_accumulation(self, tracker):
        """多次记录应正确累计"""
        tracker.record("deepseek-chat", 100, 50, 0, 0.001)
        tracker.record("deepseek-chat", 200, 100, 50, 0.002)
        tracker.record("deepseek-chat", 150, 75, 30, 0.003)

        daily = tracker.get_daily_cost()
        assert abs(daily - 0.006) < 1e-6

    def test_daily_cost_filters_by_date(self, tracker):
        """应只累计指定日期的费用"""
        # 写入手工构造的不同日期记录
        entries = [
            {"date": "2024-04-09", "usd_cost": 0.01, "model": "test",
             "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
             "timestamp": "2024-04-09T10:00:00"},
            {"date": "2024-04-10", "usd_cost": 0.02, "model": "test",
             "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
             "timestamp": "2024-04-10T10:00:00"},
        ]
        with open(tracker.cost_log_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        assert abs(
            tracker.get_daily_cost(date(2024, 4, 9)) - 0.01
        ) < 1e-6
        assert abs(
            tracker.get_daily_cost(date(2024, 4, 10)) - 0.02
        ) < 1e-6


# ===== 日报摘要测试 =====

class TestDailySummary:
    """验证日报摘要"""

    def test_summary_empty(self, tracker):
        """无记录时返回零值摘要"""
        summary = tracker.get_daily_summary()
        assert summary["call_count"] == 0
        assert summary["total_cost_usd"] == 0.0

    def test_summary_correct_counts(self, tracker):
        """摘要应包含正确的统计数据"""
        tracker.record("deepseek-chat", 100, 50, 20, 0.001)
        tracker.record("deepseek-chat", 200, 100, 80, 0.002)

        summary = tracker.get_daily_summary()
        assert summary["call_count"] == 2
        assert summary["total_prompt_tokens"] == 300
        assert summary["total_completion_tokens"] == 150
        assert summary["total_cache_hit_tokens"] == 100
        assert abs(summary["total_cost_usd"] - 0.003) < 1e-6


# ===== 月度汇总测试 =====

class TestMonthlySummary:
    """验证月度汇总"""

    def test_monthly_summary(self, tracker):
        """月度汇总应按月份聚合"""
        entries = [
            {"date": "2024-03-15", "usd_cost": 0.01, "model": "test",
             "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
             "timestamp": "2024-03-15T10:00:00"},
            {"date": "2024-03-20", "usd_cost": 0.02, "model": "test",
             "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
             "timestamp": "2024-03-20T10:00:00"},
            {"date": "2024-04-01", "usd_cost": 0.05, "model": "test",
             "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
             "timestamp": "2024-04-01T10:00:00"},
        ]
        with open(tracker.cost_log_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        monthly = tracker.update_monthly_summary()
        assert abs(monthly.get("2024-03", 0) - 0.03) < 1e-6
        assert abs(monthly.get("2024-04", 0) - 0.05) < 1e-6

    def test_monthly_summary_saved_to_file(self, tracker):
        """月度汇总应保存到 cost_summary.json"""
        tracker.record("test", 100, 50, 0, 0.01)
        tracker.update_monthly_summary()

        assert tracker.cost_summary_path.exists()
        data = json.loads(tracker.cost_summary_path.read_text())
        assert "monthly_cost_usd" in data
