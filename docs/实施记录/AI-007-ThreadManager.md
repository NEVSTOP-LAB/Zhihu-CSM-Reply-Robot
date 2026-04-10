# AI-007 实施记录：ThreadManager — 对话线程管理

## 任务目标
实现线程文件的创建、读取和追加。

## 实施内容

### 1. 模块实现 (`scripts/thread_manager.py`)
- **get_or_create_thread()**: 创建或获取 `archive/articles/{id}/threads/{thread_id}.md`
- **append_turn()**: 支持4种类型的对话追加
  - 用户首评（引用格式 `>`）
  - 用户追问（追问标记）
  - 机器人回复（含 model + tokens 信息）
  - 真人回复（⭐ 标记 + human_replied=true 更新）
- **build_context_messages()**: 解析对话内容，转换为 OpenAI messages 格式，支持 max_turns 截断

### 2. 文件格式
- YAML front-matter: thread_id, article_id, commenter, turn_count, human_replied 等
- Markdown 正文: 按时间顺序追加对话记录，`---` 分隔

## 测试结果
```
tests/test_thread_manager.py — 16 项测试全部通过 ✅
- TestGetOrCreateThread: 4 项（新建、复用、front-matter解析、不同评论不同线程）
- TestAppendTurn: 6 项（用户评论、机器人回复、真人⭐标记、front-matter更新、多次解析、计数递增）
- TestBuildContextMessages: 6 项（格式正确、user role、assistant role、截断、空线程、真人=assistant）
```

## 验收状态
✅ 单元测试全部通过
