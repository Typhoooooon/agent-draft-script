"""
agent.py —— Agent 核心引擎
===========================
实现 ReAct 循环：Think → Act → Observe → Think → ...

这是整个项目的核心文件。你可以单独运行它来测试最简单的 Agent：
    python agent.py
"""

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools import execute_tool, get_tool_schemas, list_tools

load_dotenv()

# ═══════════════════════════════════════════════════════════
# LLM 客户端（延迟初始化，避免 import 时就需要 API key）
# ═══════════════════════════════════════════════════════════

_client: OpenAI | None = None


def _get_client() -> OpenAI:  # 定义一个获取OpenAI客户端的函数，返回类型为OpenAI
    global _client  # 声明使用全局变量_client
    if _client is None:  # 检查_client是否为None
        _client = OpenAI(  # 如果_client为None，则创建一个新的OpenAI客户端实例
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量中获取API密钥
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),  # 从环境变量中获取基础URL，如果未设置则使用默认值
        )
    return _client  # 返回客户端实例


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """Agent 运行配置"""
    model: str = "deepseek-chat"
    max_steps: int = 15          # 最多执行多少轮
    max_consecutive_errors: int = 3  # 连续出错N次后终止
    verbose: bool = True         # 是否打印每个步骤的细节


# ═══════════════════════════════════════════════════════════
# LLM 决策：直接回答 or 调工具？
# ═══════════════════════════════════════════════════════════

def llm_decide(
    messages: list[dict],
    model: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    让 LLM 看到对话历史 + 可用工具，决定下一步做什么。

    Returns:
        {"type": "text", "content": "..."}
        {"type": "tool_call", "name": "...", "id": "...", "arguments": {...}}
    """
    full_messages = []

    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})

    full_messages.extend(messages)

    response = _get_client().chat.completions.create(
        model=model,
        messages=full_messages,
        tools=get_tool_schemas() if get_tool_schemas() else None,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    # 情况 1：LLM 要调用工具（优先检查——即使同时有文本也优先执行工具）
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return {
            "type": "tool_call",
            "name": tc.function.name,
            "id": tc.id,
            "arguments": json.loads(tc.function.arguments),
        }

    # 情况 2：LLM 直接输出文本（不再需要工具）
    if msg.content:
        return {"type": "text", "content": msg.content}

    return {"type": "text", "content": "（LLM 未返回有效决策）"}


# ═══════════════════════════════════════════════════════════
# ReAct 主循环
# ═══════════════════════════════════════════════════════════

def agent_run(
    user_query: str,
    config: AgentConfig | None = None,
    system_prompt: str | None = None,
) -> str:
    """
    执行完整的 Agent ReAct 循环。

    参数:
        user_query:   用户的输入问题
        config:       Agent 运行配置
        system_prompt: 可选的系统级提示词

    返回:
        Agent 的最终回答文本

    ReAct 循环流程:
        用户输入 → [对话历史]
            ↓
        ┌───────────────────────────┐
        │  LLM 决策                   │
        │  直接回答? → 返回给用户       │
        │  调工具?   → 执行工具 → 观察  │
        │             → 结果加入历史    │
        │             → 回到 LLM 决策   │
        └───────────────────────────┘
    """
    if config is None:
        config = AgentConfig()

    log = print if config.verbose else lambda *a, **kw: None

    log("=" * 60)
    log(f"🚀 Agent 启动")
    log(f"📝 用户问题: {user_query}")
    log(f"🔧 可用工具: {', '.join(t['function']['name'] for t in get_tool_schemas() if 'function' in t)}")
    log("=" * 60)

    # 初始化历史：用户原始问题始终保留在第一条
    history: list[dict] = [{"role": "user", "content": user_query}]
    consecutive_errors = 0

    for step in range(1, config.max_steps + 1):
        log(f"\n{'─' * 60}")
        log(f"📍 Step {step}/{config.max_steps}")

        # ─── 让 LLM 做决策 ───
        decision = llm_decide(
            messages=history,
            model=config.model,
            system_prompt=system_prompt,
        )

        # ─── 情况 A：直接回答 → 结束 ───
        if decision["type"] == "text":
            log(f"✅ 最终回答:\n{decision['content']}")
            log(f"\n{'=' * 60}")
            log(f"🏁 Agent 完成 (共 {step} 步)")
            log(f"{'=' * 60}")
            return decision["content"]

        # ─── 情况 B：调用工具 → 执行 → 观察 → 继续 ───
        if decision["type"] == "tool_call":
            tool_name = decision["name"]
            tool_args = decision["arguments"]
            tool_id = decision["id"]

            log(f"🔧 调用: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

            # 执行工具
            result = execute_tool(tool_name, **tool_args)

            if result.startswith("❌") or result.startswith("错误"):
                consecutive_errors += 1
                log(f"⚠️  结果: {result} (连续错误: {consecutive_errors})")
                if consecutive_errors >= config.max_consecutive_errors:
                    log(f"🛑 连续错误达到上限，终止")
                    return f"Agent 终止：连续 {consecutive_errors} 次工具调用出错。\n最后一次错误: {result}"
            else:
                consecutive_errors = 0
                log(f"👁 结果: {result[:200]}{'...' if len(result) > 200 else ''}")

            # 将 tool_call + tool_result 写入对话历史
            history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args, ensure_ascii=False),
                    },
                }],
            })
            history.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })
            continue

    log(f"\n{'=' * 60}")
    log(f"🛑 Agent 达到最大步数限制 ({config.max_steps}步)")
    log(f"{'=' * 60}")

    # 步数用完，强制让 LLM 给个最终回答
    final = llm_decide(
        messages=history
        + [
            {
                "role": "user",
                "content": "已达到最大步数限制。请基于现有信息给出你的最佳回答。",
            }
        ],
        model=config.model,
        system_prompt=system_prompt,
    )
    return final.get("content", "Agent 未能完成该任务。")


# ═══════════════════════════════════════════════════════════
# 简单封装：单轮问答（无工具，纯 LLM）
# ═══════════════════════════════════════════════════════════

def pure_llm(question: str, model: str = "deepseek-chat") -> str:
    """纯 LLM 问答——没有工具、没有循环。用于对比。"""
    resp = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(list_tools())
    print()

    # 简单测试
    query = "北京今天天气怎么样？用中文回答。"

    print("─── 对比：纯 LLM（无工具）───")
    print(pure_llm(query))
    print()

    print("─── Agent（有工具）───")
    result = agent_run(query)
