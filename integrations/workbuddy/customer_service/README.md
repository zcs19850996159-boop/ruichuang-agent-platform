# WorkBuddy 接入

将本目录作为 Skill 安装，并确保 Python 的模块搜索路径包含项目的 `work/` 目录。

默认 MCP 仅暴露 `answer_customer_question`，避免普通客服请求被通用 Agent 拆成多轮工具调用。知识摄取、评测、发布和回滚使用独立管理入口，并需要单独的管理员凭据。

协议基线：MCP `2025-11-25`，stdio 传输。
