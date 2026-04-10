# AI-003 实施记录：ZhihuClient — 读取与写入接口

## 任务目标
实现知乎评论读取与发布，含认证、分页、CSRF、限流处理。

## 实施内容

### 1. 模块实现 (`scripts/zhihu_client.py`)
- **ZhihuClient(cookie)**: 从 Cookie 字符串初始化，自动提取 `_xsrf` CSRF token
- **get_comments(object_id, object_type, since_id)**: 
  - 支持 `article` 和 `question` 两种类型
  - 自动分页（`is_end=True` 停止）
  - 请求间随机延迟 1~2 秒
  - 429 指数退避重试最多 3 次
- **post_comment(object_id, object_type, content, parent_id)**:
  - 调用 `POST https://api.zhihu.com/v4/comments`
  - 从 Cookie 提取 `_xsrf` 设置 `x-xsrftoken` 请求头
  - 发布失败返回 False（不抛异常，由主流程处理）
- **Comment dataclass**: id, parent_id, content, author, created_time, is_author_reply
- **异常类**: ZhihuAuthError (401/403), ZhihuRateLimitError (429 重试耗尽)

### 2. 设计决策
- 浏览器指纹（UA、Referer、sec-ch-ua）参考 zhihu-cli
- 写操作失败时不抛出认证异常，改为返回 False，与 pending/ 模式配合
- _xsrf 提取使用正则匹配，支持各种 Cookie 格式

## 测试结果
```
tests/test_zhihu_client.py — 22 项测试全部通过 ✅
- TestZhihuClientInit: 3 项（正常初始化、空Cookie、缺_xsrf）
- TestCommentParsing: 4 项（普通评论、追问、作者回复、字段检查）
- TestGetComments: 5 项（article分页、question端点、多页、since_id过滤、无效type）
- TestErrorHandling: 4 项（401、403、429重试成功、429全部耗尽）
- TestPostComment: 6 项（目标URL、CSRF头、payload、成功返回、失败返回、无parent_id）
```

## 验收状态
✅ 单元测试全部通过
