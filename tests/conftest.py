"""
AI-002: pytest 全局配置与共享 fixtures
参考: docs/plan/README.md § AI-002

提供测试中常用的配置加载、临时目录等共享 fixtures。
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# 确保 scripts/ 目录在 import 路径中
ROOT_DIR = Path(__file__).resolve().parent.parent
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
def articles_config(config_dir: Path) -> dict:
    """加载 config/articles.yaml 并返回解析后的字典"""
    with open(config_dir / "articles.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def settings_config(config_dir: Path) -> dict:
    """加载 config/settings.yaml 并返回解析后的字典"""
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_dir():
    """提供一个临时目录，测试结束后自动清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
