# 智能体 RESTful API 接口说明

## 1. 基础接口信息

| 配置项 | 标准值 | 客服场景说明 |
| --- | --- | --- |
| 接口类型 | RESTful API | 无状态 HTTP 入口，适配高并发和分布式部署 |
| 核心端点 | `/chat` | 唯一客服交互入口，兼容文本与图片咨询 |
| 请求方式 | `POST` | 仅支持 POST，保证长文本和 Base64 图片传输完整性 |
| 通信协议 | HTTP/1.1（测试）、HTTPS（生产） | 生产环境建议强制 HTTPS |
| 字符编码 | UTF-8 | 统一支持中文、英文、标点和特殊符号 |
| 认证方式 | Bearer Token | 生产环境设置 `KAFU_API_TOKEN` 后强制认证 |

## 2. 认证规范

所有生产请求必须在 HTTP Header 中携带认证令牌：

```http
Authorization: Bearer {KAFU_API_TOKEN}
```

本地开发环境如果未配置 `KAFU_API_TOKEN`，接口默认允许免认证调试。部署给评审或第三方系统时，应在 `.env` 中配置：

```bash
KAFU_API_TOKEN=your_token_here
API_AUTH_REQUIRED=1
```

## 3. `/chat` 请求规范

### 3.1 请求头

| 字段名 | 必选 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `Content-Type` | 是 | String | 固定 JSON 格式 | `application/json; charset=utf-8` |
| `Authorization` | 生产必选 | String | Bearer Token 认证 | `Bearer sk_customer_xxx` |
| `X-Request-Id` | 否 | String | 请求唯一标识，用于问题追溯 | `kf_req_123e4567` |
| `X-Client-Type` | 否 | String | 调用方终端 | `web` / `app` / `wx_miniprogram` |

### 3.2 请求体

| 字段名 | 必选 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `question` | 是 | String | - | 用户的客服问题字符串，长度至少 1 |
| `images` | 否 | String[] | `[]` | Base64 图片列表，支持 0-3 张，每张解码后不超过 5MB |
| `session_id` | 否 | String | 自动生成 | 客服会话 ID，用于多轮追问和上下文隔离 |
| `stream` | 否 | Boolean | `false` | 当前版本同步返回完整答案；暂不启用流式响应 |

图片字段必须使用完整 Data URL 前缀：

```text
data:image/{png/jpg/jpeg/webp};base64,{编码内容}
```

`question` 中也可以包含公网 HTTP/HTTPS 图片链接。服务会在认证通过后安全解析链接、阻止内网和非标准端口访问，并把有效图片送入视觉理解链路。当前视频页面会提取公开封面作为降级视觉输入；若封面也无法获取，响应中的 `remote_media.errors` 会说明原因，系统不会根据不可见媒体猜测答案。

### 3.3 请求示例

极简文本调用：

```json
{
  "question": "我想更换健身追踪器的表带，有其他尺寸可选吗？"
}
```

多模态调用：

```json
{
  "question": "物流一直显示待揽收，是什么原因？",
  "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."],
  "session_id": "kf_session_889900",
  "stream": false
}
```

## 4. 响应规范

成功响应体：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "智能体返回的客服答案",
    "session_id": "kf_session_889900",
    "timestamp": 1741008000
  }
}
```

为便于比赛验证和工程调试，当前实现会额外返回以下字段：

| 字段 | 说明 |
| --- | --- |
| `images` | 答案中引用的手册图片 ID 列表 |
| `ret` | 比赛提交格式：`"答案", ["图片ID"]` |
| `route` | 路由结果，如 `manual`、`policy_service`、`out_of_scope` |
| `selector` | 图片选择器的候选和置信度信息 |
| `memory` | 多轮会话解析结果 |
| `input_images` | 上传图片接收数量、可用数量、是否参与视觉理解 |
| `remote_media` | 问题中外链媒体的检测数量、可用数量、解析类型和错误原因 |
| `sources` | 标准化证据列表，包含手册、章节、chunk、图片、证据片段、版本 metadata 和各阶段分数 |
| `retrieval` | embedding、BM25、旧规则融合后的 rerank 分数、Top 分差、阈值、图片截断数和决策 |
| `refusal_type` | 拒答/澄清类型；正常回答为空，可为 `product_unclear`、`evidence_insufficient` 等 |
| `answer_check` | `<PIC>` 数量、图片数量和约束校验结果 |
| `elapsed_ms` | 服务端处理耗时 |
| `request_id` | 若请求头传入 `X-Request-Id`，响应中会回传 |
| `client_type` | 若请求头传入 `X-Client-Type`，响应中会回传 |

错误响应体：

```json
{
  "code": 400,
  "msg": "question/text is required",
  "data": null
}
```

常见状态码：

| 状态码 | 说明 |
| --- | --- |
| 400 | 请求体、字段或 Base64 图片格式错误 |
| 401 | Bearer Token 缺失或错误 |
| 404 | 路径不存在 |
| 413 | 请求体过大 |
| 415 | `Content-Type` 不是 JSON |
| 500 | 服务端内部错误 |

## 5. 调用示例

PowerShell：

```powershell
$headers = @{
  "Content-Type" = "application/json; charset=utf-8"
  "Authorization" = "Bearer $env:KAFU_API_TOKEN"
  "X-Request-Id" = "kf_req_demo_001"
  "X-Client-Type" = "web"
}

$body = @{
  question = "空调遥控器没电了，按照手册应该怎样更换电池？"
  images = @()
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://127.0.0.1:8765/chat" -Method Post -Headers $headers -Body $body
```

curl：

```bash
curl -X POST "http://127.0.0.1:8765/chat" \
  -H "Content-Type: application/json; charset=utf-8" \
  -H "Authorization: Bearer ${KAFU_API_TOKEN}" \
  -H "X-Request-Id: kf_req_demo_001" \
  -H "X-Client-Type: web" \
  -d '{"question":"空调遥控器没电了，按照手册应该怎样更换电池？","images":[]}'
```

## 6. 多模态与多轮说明

图片不会替代知识库证据。上传图片后，系统先调用视觉模型生成“可见事实摘要”，再将该摘要作为检索和意图识别的补充信息；最终答案仍由手册 RAG、客服政策和图片选择器约束。

手册检索采用多语言 E5 embedding、BM25 和原有规则分数融合 rerank。`retrieval.top_rerank_score` 与
`retrieval.top_score_gap` 用于质检；证据低于阈值时不会继续自由生成，而会返回细分的澄清类型。每次最多返回
8 张手册图片，截断数量记录在 `retrieval.images_truncated`。索引中的 chunk 和图片 caption 均带
`source_hash`、`chunk_version`、`section_path`、`source_page`、`language`、`review_status`，支持后续按 hash
增量重建与知识版本追踪。

`session_id` 是多轮对话的隔离单位。同一 `session_id` 会继承上一轮的产品、手册和政策主题；不同用户、不同任务或点击“新建聊天”时应使用新的 `session_id`。
