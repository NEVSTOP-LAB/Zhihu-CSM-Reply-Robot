# AI-008 实施记录：CommentFilter — 前置过滤器

## 任务目标
实现评论过滤（广告/重复/超长截断）。

## 实施内容

### 1. 模块实现 (`scripts/comment_filter.py`)
- **should_skip()**: 按优先级判断是否跳过
  - 广告关键词（不区分大小写）
  - 重复评论（同一作者在 dedup_window_minutes 内重复）
- **truncate_comment()**: 超长截断（tiktoken 在线 / 字符估算离线降级）
- **reset_dedup_cache()**: 用于测试和新一天开始时重置

### 2. 遇到的问题
- **tiktoken 网络依赖**: tiktoken.get_encoding() 需要下载编码文件，在受限网络环境中会失败
  - 解决方案: 添加 try/except 降级到字符估算模式（1 token ≈ 0.67 中文字符）

## 测试结果
```
tests/test_comment_filter.py — 15 项测试全部通过 ✅
- TestSpamFilter: 4 项（关键词命中、大小写不敏感、多种关键词、正常内容通过）
- TestDedupFilter: 4 项（首条通过、窗口内重复跳过、超窗口通过、不同用户独立）
- TestTruncation: 4 项（短评论不截断、长评论截断、前部保留、截断不跳过）
- TestIntegration: 3 项（正常评论全通过、广告优先于重复、空配置不过滤）
```

## 验收状态
✅ 单元测试全部通过
