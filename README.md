# Mini-Coding-Agent

> 基于 **LangGraph** 的多智能体编码助手，支持 Planner → Coder → Tester → Reviewer 四角色协作与实时可视化监控。

## ✨ 项目亮点

- **🧠 多智能体流水线（Multi-Agent Pipeline）** — 基于 LangGraph 编排 Planner、Coder、Tester、Reviewer 四角色，支持自动反馈循环（feedback loop），实现「规划 → 编码 → 测试 → 评审」的闭环协作。
- **📊 实时监控面板（Real-Time Dashboard）** — 基于 FastAPI + WebSocket 的 Web UI，可实时观察各 Agent 的执行状态、节点高亮、事件流推送。
- **🔧 角色可配置（Configurable Roles）** — 通过 YAML 定义各角色的身份（identity）、工具权限与策略，无需修改代码即可调整 Prompt 或替换模型。
- **🤖 多后端模型支持** — 支持 Ollama（本地）及任意 OpenAI 兼容 API（OpenAI、Moonshot、DeepSeek 等），可一键切换。
- **🛡️ 容错解析** — 自动修复模型输出的畸形 JSON/XML 工具调用；支持单轮多 tool 输出截断保护。
- **💾 会话持久化** — 单智能体会话支持断点续传，从磁盘恢复历史上下文。
- **🔒 权限管控** — 针对 `write_file`、`run_shell` 等高危工具，支持 `ask` / `auto` / `never` 三级审批策略。

## 🏗 系统架构

```
┌─────────────┐     ┌──────────┐     ┌────────┐     ┌────────┐     ┌──────────┐
│   Browser   │◄────┤ FastAPI  │────►│ Event  │────►│LangGraph│────►│  LLM   │
│  Dashboard  │ WS  │  Server  │     │  Bus   │     │ Workflow│     │(Ollama │
└─────────────┘     └──────────┘     └────────┘     └────────┘     │ / API) │
                                                                      └────────┘
                                                                         │
                                    ┌────────────────────────────────────┘
                                    ▼
       ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
       │ START  │───►│Planner │───►│ Coder  │───►│ Tester │───►│Reviewer│
       └────────┘    └────────┘    └────────┘    └────────┘    └───┬────┘
                                                                    │
                     ┌──────────────────────────────────────────────┤
                     │                                              │
                     ▼                                              ▼
              ┌────────────┐                                 ┌────────┐
              │    END     │                                 │ Coder  │
              │(approved / │                                 │(needs_│
              │ max_cycles)│                                 │ fix)   │
              └────────────┘                                 └────────┘
```

## 🚀 快速开始

### 1. 安装依赖

**方式一：使用 conda + pip（常用）**

```bash
# 创建并激活虚拟环境
conda create -n mca python=3.12
conda activate mca

# 安装运行时依赖
pip install -r requirements.txt

# 开发调试时，可额外安装开发依赖
pip install -r requirements-dev.txt
```

**方式二：使用 uv（推荐，更快）**

```bash
uv pip install -e .
```

**方式三：直接使用 pip**

```bash
pip install -e .
```

### 2. 配置模型

编辑 `config/default.yaml`：

```yaml
model:
  name: "gpt-4o-mini"
  api_key: "sk-your-api-key"
  base_url: "https://api.openai.com/v1"
  temperature: 0.2
  max_new_tokens: 2048

agent:
  approval_policy: "auto"
```

> 本地模型使用 Ollama：
> ```yaml
> model:
>   name: "qwen3.5:4b"
>   host: "http://127.0.0.1:11434"
> ```

### 3. 启动监控面板

```bash
python -m mini_coding_agent --serve
```

打开 http://localhost:8080，输入需求后点击 **运行**。

### 4. 命令行模式（单次运行）

```bash
python -m mini_coding_agent \
  --mode multi \
  --approval auto \
  "请实现一个快速排序算法"
```

## 🎛 配置说明

```yaml
model:
  name: "gpt-4o-mini"           # 模型标识
  api_key: ""                   # API Key（也可通过环境变量 OPENAI_API_KEY 传入）
  base_url: "https://api.openai.com/v1"
  temperature: 0.2
  top_p: 0.9
  timeout: 300
  max_new_tokens: 2048

agent:
  max_steps: 10                  # 每轮请求单个 Agent 的最大工具调用次数
  approval_policy: "auto"        # ask | auto | never

multi_agent:
  max_steps_planner: 10
  max_steps_coder: 10
  max_steps_tester: 8
  max_steps_reviewer: 6
  max_review_cycles: 5           # Coder↔Reviewer 最大反馈轮数

roles:
  planner:
    identity: "You are a Software Architect Planner..."
  coder:
    identity: "You are an expert Software Engineer..."
```

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| 工作流引擎 | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Web 服务 | [FastAPI](https://fastapi.tiangolo.com/) + WebSocket |
| 前端 | Vanilla JS + CSS（零构建） |
| LLM 接入 | Ollama (`/api/generate`) + OpenAI 兼容 API (`/chat/completions`) |
| 配置管理 | YAML |
| 测试 | pytest |

## 🧪 运行测试

```bash
pytest tests/ -q
```

## 📢 声明

本项目受 [mini-coding-agent](https://github.com/rasbt/mini-coding-agent) 启发，在此基础上增加了多智能体编排、实时可观测性与可插拔后端支持。
