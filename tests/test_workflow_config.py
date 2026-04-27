# -*- coding: utf-8 -*-
"""
GitHub Actions Workflow 配置测试
================================

实施计划关联：AI-004 验收标准
验证 workflow YAML 文件的结构和关键字段。

测试独立于实现，直接解析 YAML 验证结构。
"""
from pathlib import Path

import pytest
import yaml


WORKFLOW_DIR = Path(__file__).parent.parent / ".github" / "workflows"


class TestBotWorkflow:
    """bot.yml 主 Workflow 测试"""

    @pytest.fixture
    def workflow(self) -> dict:
        """加载 bot.yml"""
        path = WORKFLOW_DIR / "bot.yml"
        assert path.exists(), f"Workflow 文件不存在: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_workflow_has_name(self, workflow: dict):
        """Workflow 应有名称"""
        assert "name" in workflow

    def test_has_schedule_trigger(self, workflow: dict):
        """应包含 schedule 定时触发"""
        triggers = workflow.get("on", workflow.get(True, {}))
        assert "schedule" in triggers, "缺少 schedule 触发器"

    def test_cron_expression_valid(self, workflow: dict):
        """cron 表达式应为每15分钟"""
        triggers = workflow.get("on", workflow.get(True, {}))
        schedules = triggers.get("schedule", [])
        assert len(schedules) > 0, "schedule 列表为空"
        cron = schedules[0].get("cron", "")
        assert "*/15" in cron, f"cron 应为每15分钟，实际: {cron}"

    def test_has_workflow_dispatch(self, workflow: dict):
        """应支持手动触发"""
        triggers = workflow.get("on", workflow.get(True, {}))
        assert "workflow_dispatch" in triggers, "缺少 workflow_dispatch"

    def test_has_contents_write_permission(self, workflow: dict):
        """应有 contents: write 权限（用于 git push）"""
        perms = workflow.get("permissions", {})
        assert perms.get("contents") == "write", "缺少 contents: write 权限"

    def test_has_issues_write_permission(self, workflow: dict):
        """应有 issues: write 权限（用于告警创建 Issue）"""
        perms = workflow.get("permissions", {})
        assert perms.get("issues") == "write", "缺少 issues: write 权限"

    def test_job_runs_on_ubuntu(self, workflow: dict):
        """Job 应运行在 ubuntu-latest"""
        job = workflow.get("jobs", {}).get("reply-bot", {})
        assert "ubuntu" in job.get("runs-on", ""), "应运行在 ubuntu"

    def test_has_checkout_step(self, workflow: dict):
        """应包含 checkout 步骤"""
        steps = workflow["jobs"]["reply-bot"]["steps"]
        step_uses = [s.get("uses", "") for s in steps]
        assert any("checkout" in u for u in step_uses), "缺少 checkout 步骤"

    def test_has_python_setup(self, workflow: dict):
        """应包含 Python 环境设置"""
        steps = workflow["jobs"]["reply-bot"]["steps"]
        step_uses = [s.get("uses", "") for s in steps]
        assert any("setup-python" in u for u in step_uses), "缺少 setup-python"

    def test_has_pip_cache(self, workflow: dict):
        """应包含 pip 缓存"""
        steps = workflow["jobs"]["reply-bot"]["steps"]
        step_uses = [s.get("uses", "") for s in steps]
        cache_steps = [u for u in step_uses if "cache" in u]
        assert len(cache_steps) >= 1, "缺少 pip 缓存步骤"

    def test_has_huggingface_cache(self, workflow: dict):
        """应包含 HuggingFace 模型缓存"""
        steps = workflow["jobs"]["reply-bot"]["steps"]
        step_names = [s.get("name", "").lower() for s in steps]
        assert any("huggingface" in n for n in step_names), \
            "缺少 HuggingFace 缓存步骤"

    def test_secrets_referenced(self, workflow: dict):
        """应引用必要的 Secrets"""
        workflow_str = yaml.dump(workflow)
        assert "LLM_API_KEY" in workflow_str, "缺少 LLM_API_KEY 引用"
        assert "GITHUB_TOKEN" in workflow_str, "缺少 GITHUB_TOKEN 引用"

    def test_run_bot_script_path(self, workflow: dict):
        """主脚本路径应为 scripts/run_bot.py"""
        steps = workflow["jobs"]["reply-bot"]["steps"]
        run_commands = [s.get("run", "") for s in steps]
        assert any("scripts/run_bot.py" in r for r in run_commands), \
            "主脚本路径应为 scripts/run_bot.py"

    def test_git_config_before_push(self, workflow: dict):
        """应先配置 git 身份再 push"""
        steps = workflow["jobs"]["reply-bot"]["steps"]
        run_commands = [s.get("run", "") for s in steps]
        commit_step = [r for r in run_commands if "git commit" in r]
        assert len(commit_step) > 0, "缺少 git commit 步骤"
        # 确保设置了 user.name 和 user.email
        commit_cmd = commit_step[0]
        assert "git config user.name" in commit_cmd
        assert "git config user.email" in commit_cmd

    def test_skip_ci_in_commit(self, workflow: dict):
        """commit message 应包含 [skip ci] 防止循环"""
        steps = workflow["jobs"]["reply-bot"]["steps"]
        run_commands = [s.get("run", "") for s in steps]
        commit_step = [r for r in run_commands if "git commit" in r]
        assert len(commit_step) > 0
        assert "skip ci" in commit_step[0].lower(), \
            "commit message 应包含 [skip ci]"

    def test_has_concurrency_setting(self, workflow: dict):
        """应有 concurrency 设置防止并发写 seen_ids（FIX-22）"""
        concurrency = workflow.get("concurrency", {})
        assert concurrency, "缺少 concurrency 设置（FIX-22）"
        assert "group" in concurrency, "concurrency 应包含 group"
        assert concurrency.get("cancel-in-progress") is False, (
            "cancel-in-progress 应为 false，避免正在运行的 bot 被取消"
        )


class TestSyncWikiWorkflow:
    """sync-wiki.yml Workflow 测试"""

    @pytest.fixture
    def workflow(self) -> dict:
        """加载 sync-wiki.yml"""
        path = WORKFLOW_DIR / "sync-wiki.yml"
        assert path.exists(), f"Workflow 文件不存在: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_has_weekly_schedule(self, workflow: dict):
        """应有每周触发的 schedule"""
        triggers = workflow.get("on", workflow.get(True, {}))
        schedules = triggers.get("schedule", [])
        assert len(schedules) > 0
        # 验证是周日运行
        cron = schedules[0].get("cron", "")
        assert "0" in cron.split()[-1] or "0" in cron, \
            f"应为每周日运行，实际: {cron}"

    def test_has_force_rebuild_input(self, workflow: dict):
        """应支持手动触发并带强制重建参数"""
        triggers = workflow.get("on", workflow.get(True, {}))
        dispatch = triggers.get("workflow_dispatch", {})
        inputs = dispatch.get("inputs", {})
        assert "force_rebuild" in inputs, "缺少 force_rebuild 输入参数"


class TestCacheKeyStrategy:
    """验证 vector store cache key 使用文件哈希而非 run_number（FIX-08）"""

    def _get_vector_cache_key(self, workflow: dict) -> str:
        """从 workflow 中提取 vector stores 的 cache key"""
        steps = workflow["jobs"]
        for job in steps.values():
            for step in job.get("steps", []):
                with_cfg = step.get("with", {})
                path = with_cfg.get("path", "")
                if "vector_store" in str(path):
                    return with_cfg.get("key", "")
        return ""

    def test_bot_vector_cache_key_not_run_number(self):
        """bot.yml vector stores cache key 不应使用 run_number（FIX-08）"""
        path = WORKFLOW_DIR / "bot.yml"
        with open(path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
        key = self._get_vector_cache_key(workflow)
        assert key, "未找到 vector_store cache step"
        assert "run_number" not in key, (
            f"cache key 不应使用 run_number，当前: {key}"
        )

    def test_bot_vector_cache_key_uses_hash(self):
        """bot.yml vector stores cache key 应基于文件哈希（FIX-08）"""
        path = WORKFLOW_DIR / "bot.yml"
        with open(path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
        key = self._get_vector_cache_key(workflow)
        assert "hashFiles" in key, (
            f"cache key 应使用 hashFiles()，当前: {key}"
        )

    def test_sync_wiki_vector_cache_key_uses_hash(self):
        """sync-wiki.yml vector stores cache key 应基于文件哈希（FIX-08）"""
        path = WORKFLOW_DIR / "sync-wiki.yml"
        with open(path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
        key = self._get_vector_cache_key(workflow)
        assert "hashFiles" in key, (
            f"cache key 应使用 hashFiles()，当前: {key}"
        )
