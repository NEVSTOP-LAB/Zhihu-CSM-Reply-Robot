# AI-004 实施记录：GitHub Actions Workflow 基础版

## 任务目标
创建可运行的 workflow，含 HuggingFace 模型缓存。

## 实施内容

### 1. Workflow 文件 (`.github/workflows/bot.yml`)
- **触发条件**: cron `0 2,8,14,20 * * *`（每6小时） + `workflow_dispatch`（手动）
- **权限**: `contents: write`（推送存档）+ `issues: write`（创建告警）
- **缓存**: pip + HuggingFace 模型 + 向量库（Actions Cache）
- **Secrets**: ZHIHU_COOKIE, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, GITHUB_TOKEN
- **Git 提交**: 配置 bot 身份，`[skip ci]` 防止循环触发

### 2. 遇到的问题
- **PyYAML `on` 解析问题**: YAML 的 `on` 键被 PyYAML 解析为布尔值 `True`，需在测试中特殊处理
  - 解决方案：添加 `_get_triggers()` 辅助函数，同时检查 `"on"` 和 `True` 键

## 测试结果
```
tests/test_workflow_config.py — 14 项测试全部通过 ✅
- TestWorkflowTriggers: 3 项（schedule、cron表达式、workflow_dispatch）
- TestWorkflowPermissions: 2 项（contents:write、issues:write）
- TestWorkflowSteps: 7 项（job存在、ubuntu、checkout、python、pip缓存、HF缓存、run_bot）
- TestWorkflowSecrets: 2 项（secrets引用、git commit [skip ci]）
```

## 验收状态
✅ 测试全部通过
