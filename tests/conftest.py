# -*- coding: utf-8 -*-
"""
测试公共 fixtures
================

参考：docs/plan/README.md AI-002

提供所有测试模块共用的 pytest fixtures，包括：
- 临时目录创建
- 配置文件路径
- Mock 对象工厂
"""
import os
import sys
from pathlib import Path

import pytest

# 确保 scripts 目录在导入路径中
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def project_root() -> Path:
    """返回项目根目录路径"""
    return ROOT_DIR


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """返回 config/ 目录路径"""
    return project_root / "config"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """创建临时 data/ 目录，用于测试数据存储"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def tmp_archive_dir(tmp_path: Path) -> Path:
    """创建临时 archive/ 目录，用于测试归档"""
    archive_dir = tmp_path / "archive" / "articles"
    archive_dir.mkdir(parents=True)
    return archive_dir


@pytest.fixture
def tmp_pending_dir(tmp_path: Path) -> Path:
    """创建临时 pending/ 目录，用于测试待审核文件"""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir(parents=True)
    return pending_dir
