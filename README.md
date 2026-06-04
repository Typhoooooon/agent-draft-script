# 🤖 Agent 从零实现

> 教学级 AI Agent 框架——从纯 LLM 到多 Skill 协作，逐层讲解每一步在做什么。

## 项目结构

```
agent_from_scratch/
├── tools.py          # Tool 系统：工具注册、描述、执行
├── agent.py          # Agent 核心：ReAct 循环、LLM 决策
├── skills.py         # Skill 封装：领域知识 + 工具组合
├── multi_skill.py    # 多 Skill 编排器
├── demo.py           # 5 个演示场景
├── requirements.txt  # 依赖
├── .env.example      # API Key 模板
└── README.md         # 本文件
```

## 架构概览

```
┌─────────────────────────────────┐
│         Skill（能力包）           │
│   多个 Tool + 领域知识 + 策略     │
├─────────────────────────────────┤
│         Tool（原子操作）          │
│   search_web / run_python / ... │
├─────────────────────────────────┤
│     Agent 主循环（ReAct）        │
│   Think → Act → Observe → ...  │
├─────────────────────────────────┤
│         LLM（推理引擎）           │
│         DeepSeek-Chat           │
└─────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制配置模板
copy .env.example .env

# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> 获取 Key: https://platform.deepseek.com/api_keys

### 3. 运行演示

```bash
python demo.py
```

然后选择你想看的演示场景。

### 4. 单独测试各个模块

```bash
# 测试工具系统
python tools.py

# 测试 Agent 核心循环
python agent.py

# 测试 Skill 系统
python skills.py

# 测试多 Skill 编排
python multi_skill.py
```

## 演示说明

| Demo | 内容 | 学到什么 |
|------|------|----------|
| Demo 1 | 纯 LLM vs Agent | Agent 的本质区别：能否调用外部工具 |
| Demo 2 | 多工具组合 | Agent 自动决定需要哪些工具、什么顺序 |
| Demo 3 | 单 Skill | Skill 如何提升输出质量和稳定性 |
| Demo 4 | 编程助手 | Skill 的自纠错循环 |
| Demo 5 | 多 Skill 协作 | 完整的多 Skill 编排流程 |
| Demo 6 | JD 专项 Skill | MCP 协议、多模态 AI、CV 基础学习 |

## 核心代码阅读顺序

想理解 Agent 的代码实现，按这个顺序读：

1. **`tools.py`** —— 看 Tool 怎么定义、注册、执行
2. **`agent.py`** —— 看 ReAct 循环的完整实现（核心！）
3. **`skills.py`** —— 看 Skill 怎么封装和注入
4. **`multi_skill.py`** —— 看多 Skill 怎么编排
5. **`demo.py`** —— 看完整的使用场景
