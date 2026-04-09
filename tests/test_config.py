# -*- coding: utf-8 -*-
"""
配置文件测试
============

实施计划关联：AI-002 验收标准
验证 config/articles.yaml 和 config/settings.yaml 可被正确加载，
必填字段存在且格式正确。

测试独立于实现，直接读取并验证 YAML 文件内容。
"""
from pathlib import Path

import pytest
import yaml


class TestArticlesConfig:
    """articles.yaml 配置文件测试"""

    @pytest.fixture
    def articles_config(self, config_dir: Path) -> dict:
        """加载 articles.yaml"""
        config_path = config_dir / "articles.yaml"
        assert config_path.exists(), f"配置文件不存在: {config_path}"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_articles_yaml_loads_successfully(self, articles_config: dict):
        """articles.yaml 可以被成功加载为字典"""
        assert isinstance(articles_config, dict)

    def test_articles_key_exists(self, articles_config: dict):
        """顶层必须包含 'articles' 键"""
        assert "articles" in articles_config

    def test_articles_is_list(self, articles_config: dict):
        """articles 值必须是列表"""
        assert isinstance(articles_config["articles"], list)

    def test_articles_not_empty(self, articles_config: dict):
        """articles 列表不能为空（至少有一个监控目标）"""
        assert len(articles_config["articles"]) > 0

    def test_article_required_fields(self, articles_config: dict):
        """每个文章条目必须包含 id, title, url, type 字段"""
        required_fields = {"id", "title", "url", "type"}
        for i, article in enumerate(articles_config["articles"]):
            missing = required_fields - set(article.keys())
            assert not missing, (
                f"文章条目 [{i}] 缺少必填字段: {missing}"
            )

    def test_article_type_valid(self, articles_config: dict):
        """type 字段只能是 'article' 或 'question'"""
        valid_types = {"article", "question"}
        for i, article in enumerate(articles_config["articles"]):
            assert article["type"] in valid_types, (
                f"文章条目 [{i}] 的 type 值 '{article['type']}' "
                f"不合法，应为 {valid_types}"
            )


class TestSettingsConfig:
    """settings.yaml 配置文件测试"""

    @pytest.fixture
    def settings_config(self, config_dir: Path) -> dict:
        """加载 settings.yaml"""
        config_path = config_dir / "settings.yaml"
        assert config_path.exists(), f"配置文件不存在: {config_path}"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_settings_yaml_loads_successfully(self, settings_config: dict):
        """settings.yaml 可以被成功加载为字典"""
        assert isinstance(settings_config, dict)

    def test_top_level_sections_exist(self, settings_config: dict):
        """顶层必须包含所有必要的配置区块"""
        required_sections = {"bot", "llm", "rag", "vector_store", "review", "filter", "alerting"}
        missing = required_sections - set(settings_config.keys())
        assert not missing, f"缺少必要的配置区块: {missing}"

    def test_bot_section_fields(self, settings_config: dict):
        """bot 区块必须包含关键字段"""
        bot = settings_config["bot"]
        required = {
            "check_interval_hours",
            "max_new_comments_per_run",
            "max_new_comments_per_day",
            "llm_budget_usd_per_day",
        }
        missing = required - set(bot.keys())
        assert not missing, f"bot 区块缺少字段: {missing}"

    def test_llm_section_fields(self, settings_config: dict):
        """llm 区块必须包含关键字段"""
        llm = settings_config["llm"]
        required = {"base_url", "model", "max_tokens", "temperature"}
        missing = required - set(llm.keys())
        assert not missing, f"llm 区块缺少字段: {missing}"

    def test_rag_section_fields(self, settings_config: dict):
        """rag 区块必须包含关键字段"""
        rag = settings_config["rag"]
        required = {
            "embedding_model",
            "use_online_embedding",
            "top_k",
            "similarity_threshold",
            "history_turns",
        }
        missing = required - set(rag.keys())
        assert not missing, f"rag 区块缺少字段: {missing}"

    def test_vector_store_section_fields(self, settings_config: dict):
        """vector_store 区块必须包含关键字段"""
        vs = settings_config["vector_store"]
        required = {"backend", "max_size_mb"}
        missing = required - set(vs.keys())
        assert not missing, f"vector_store 区块缺少字段: {missing}"

    def test_filter_section_fields(self, settings_config: dict):
        """filter 区块必须包含关键字段"""
        flt = settings_config["filter"]
        required = {"max_comment_tokens", "spam_keywords", "dedup_window_minutes"}
        missing = required - set(flt.keys())
        assert not missing, f"filter 区块缺少字段: {missing}"

    def test_alerting_section_fields(self, settings_config: dict):
        """alerting 区块必须包含关键字段"""
        alerting = settings_config["alerting"]
        required = {"github_issue", "consecutive_fail_limit"}
        missing = required - set(alerting.keys())
        assert not missing, f"alerting 区块缺少字段: {missing}"

    def test_review_section_fields(self, settings_config: dict):
        """review 区块必须包含 manual_mode 字段"""
        review = settings_config["review"]
        assert "manual_mode" in review, "review 区块缺少 manual_mode 字段"

    def test_bot_limits_are_positive(self, settings_config: dict):
        """bot 区块的限制值必须为正数"""
        bot = settings_config["bot"]
        assert bot["max_new_comments_per_run"] > 0
        assert bot["max_new_comments_per_day"] > 0
        assert bot["llm_budget_usd_per_day"] > 0

    def test_rag_thresholds_valid(self, settings_config: dict):
        """rag 区块的阈值必须在合理范围内"""
        rag = settings_config["rag"]
        assert 0 < rag["similarity_threshold"] < 1.0
        assert rag["top_k"] > 0
        assert rag["history_turns"] > 0
