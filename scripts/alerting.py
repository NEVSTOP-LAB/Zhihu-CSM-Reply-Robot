# -*- coding: utf-8 -*-
"""
GitHub Issue 告警模块
=====================

实施计划关联：AI-010 告警模块 — GitHub Issue 自动创建
参考文档：docs/Review-方案/ 评审建议

功能：
- 失败告警（401/403 Cookie 失效、429 限流、预算超限）
- 幂等创建（防止重复 Issue）
- Cookie 存活状态记录
"""
