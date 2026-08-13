# 睿创多模态客服智能体：第一阶段工程说明

本阶段采用模块化单体，不拆分内部微服务。正式入口仍为：

- `POST /chat`
- `POST /chat/stream`
- `GET /health`
- `GET /metrics`

## 目录边界

`work/customer_service_core` 是新的应用与架构边界。当前比赛实现由兼容适配器承载，随后按回归门禁逐项迁移。

依赖方向：

```text
FastAPI / REST / MCP / WorkBuddy adapters
                 ↓
CustomerServiceApplication
                 ↓
Routing / Retrieval / Multimodal / Generation / Validation
                 ↓
ModelGateway / Storage / External adapters
```

内部模块不得通过 HTTP 相互调用。

当前在线性能修复把不可变的 EvidenceBuilder 与混合检索索引改为进程启动时预加载。请求时直接传入动态选图结果，不再创建 `runtime_api_selector_*.jsonl`、修改 `META_IMAGE_SELECTION_CACHE` 或串行重建知识资产。

## 版本维度

每次回答同时绑定：

- Application Version
- Knowledge Version
- Model Configuration Version
- Prompt Version
- Profile
- Tenant / Knowledge Space

比赛接口缺省使用：

```text
tenant_id=default
knowledge_space_id=competition
profile=competition
```

## 启动

1. 复制 `.env.example` 为 `.env` 并填写密钥。
2. 挂载 `assets` 和 `outputs/rag_assets`。
3. 使用运行时锁定依赖启动：

```bash
python -m pip install -r requirements-runtime.lock
bash work/start_all_services.sh
```

`requirements-lock.txt` 是冻结服务器的完整 `pip freeze`，用于取证和环境比对；其中包含 Conda 本地构建路径，不能作为容器安装清单。`requirements-runtime.lock` 才是容器使用的最小运行时依赖。

## 测试

```bash
python -m unittest discover -s work/tests_phase1 -p 'test_*.py'
python work/test_official_api_contract.py
```

## 迁移原则

1. 每次只迁移一个明确行为。
2. 每次迁移后执行官方、泛化和图片回归。
3. Competition Profile 在完成等价验证前继续使用兼容适配器。
4. Enterprise Profile 不允许请求级启用比赛补丁。

量化验收条件见 `docs/PHASE1_ACCEPTANCE_GATES.md`。
