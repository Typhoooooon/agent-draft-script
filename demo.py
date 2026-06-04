"""
demo.py —— 演示脚本
===================
运行所有演示场景，展示 Agent 从简单到复杂的完整能力。

用法:
    1. 复制 .env.example 为 .env，填入你的 DEEPSEEK_API_KEY
    2. 安装依赖: pip install -r requirements.txt
    3. 运行: python demo.py

演示场景:
    1. 纯 LLM vs Agent 对比
    2. 单 Tool Agent
    3. 单 Skill Agent
    4. 多 Skill 协作 Agent
    5. 编程助手 Skill
    6. JD 专项 Skill（MCP / 多模态 / CV）
"""

import os

from dotenv import load_dotenv

from agent import AgentConfig, agent_run, pure_llm
from multi_skill import MultiSkillOrchestrator
from skills import (
    CODING_ASSIST_SKILL,
    CV_LEARN_SKILL,
    DATA_ANALYSIS_SKILL,
    MCP_LEARN_SKILL,
    MULTIMODAL_LEARN_SKILL,
    REPORT_WRITER_SKILL,
    WEB_RESEARCH_SKILL,
    SkillRegistry,
    run_skill,
)

load_dotenv()


def banner(title: str):
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}\n")


def check_config():
    """检查 API key 是否配置"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "sk-your-api-key-here":
        print("❌ 请先在 .env 文件中设置你的 DEEPSEEK_API_KEY")
        print("   1. 复制 .env.example 为 .env")
        print("   2. 在 .env 中填入你的密钥")
        print("   获取密钥: https://platform.deepseek.com/api_keys")
        return False
    return True


# ═══════════════════════════════════════════════════════════
# Demo 1: 纯 LLM vs Agent（核心对比）
# ═══════════════════════════════════════════════════════════

def demo1_llm_vs_agent():
    banner("Demo 1: 纯 LLM vs Agent —— 核心区别")

    query = "北京今天天气怎么样？"
    print(f"❓ 问题: {query}\n")

    print("─── 纯 LLM 回答（无工具，活在沙盒里）───")
    llm_answer = pure_llm(query)
    print(f">>> {llm_answer}\n")

    print("─── Agent 回答（有 search_web 工具）───")
    agent_answer = agent_run(query)
    print(f">>> {agent_answer}\n")

    print("📌 关键区别: Agent 能调用工具获取实时信息，纯 LLM 只能'猜'或说不知道。")


# ═══════════════════════════════════════════════════════════
# Demo 2: Agent 多工具组合
# ═══════════════════════════════════════════════════════════

def demo2_multi_tool():
    banner("Demo 2: Agent 多工具组合调用")

    queries = [
        "现在几点？同时帮我算一下 1 到 100 所有质数的和。",
        "搜索一下 Python 最新版本，然后用 Python 计算 2024 年到今天过了多少天。",
    ]

    for q in queries:
        print(f"❓ 问题: {q}\n")
        answer = agent_run(q)
        print(f"\n✅ 最终答案:\n{answer}\n")
        print("-" * 60)


# ═══════════════════════════════════════════════════════════
# Demo 3: 单 Skill Agent
# ═══════════════════════════════════════════════════════════

def demo3_single_skill():
    banner("Demo 3: Skill 驱动 Agent —— 领域专业化")

    print("同一个问题，不同 Skill 会给出不同侧重的答案。\n")

    query = "介绍一下 CLIP 模型的核心原理。"

    # 不加 Skill
    print("─── 无 Skill（通用 Agent）───")
    result = agent_run(query, config=AgentConfig(verbose=False))
    print(f">>> {result}\n")

    # 加 web_research Skill
    print("─── web_research Skill ───")
    result = run_skill(WEB_RESEARCH_SKILL, query, config=AgentConfig(verbose=False))
    print(f">>> {result}\n")

    print("📌 注意: Skill 会引导 Agent 主动搜索、按结构化格式输出，质量更稳定。")


# ═══════════════════════════════════════════════════════════

def demo4_coding_skill():
    banner("Demo 4: 编程助手 Skill —— 写代码 + 自纠错 + 自动保存")

    query = (
        "写一个 Python 函数，计算斐波那契数列第 30 项。"
        "先用递归写，再用动态规划写，比较两者的运行时间。"
        "确保代码能跑通。"
    )

    print(f"❓ 问题: {query}\n")
    answer = run_skill(CODING_ASSIST_SKILL, query, config=AgentConfig(verbose=False))
    print(f"✅ 结果:\n{answer}")


# ═══════════════════════════════════════════════════════════
# Demo 5: 多 Skill 协作（杀手锏）
# ═══════════════════════════════════════════════════════════

def demo5_multi_skill():
    banner("Demo 5: 多 Skill 协作 —— 复杂任务自动化")

    registry = SkillRegistry()
    registry.register(WEB_RESEARCH_SKILL)
    registry.register(DATA_ANALYSIS_SKILL)
    registry.register(REPORT_WRITER_SKILL)

    orchestrator = MultiSkillOrchestrator(registry)

    query = (
        "调研当前主流深度学习框架（PyTorch, TensorFlow, JAX）的发展趋势，"
        "用数据分析的方法对比它们的优劣势，"
        "最后生成一份结构化的技术选型报告。"
    )

    print(f"❓ 问题: {query}\n")
    output = orchestrator.execute(query)

    print("\n📋 执行计划:", " → ".join(output["plan"]))
    print("\n" + "=" * 60)
    print("🎯 最终报告:")
    print("=" * 60)
    print(output["final_answer"])


# ═══════════════════════════════════════════════════════════
# Demo 6: JD 专项 Skill —— MCP / 多模态 / CV 基础
# ═══════════════════════════════════════════════════════════

def demo6_jd_skills():
    banner("Demo 6: JD 专项 Skill —— Agent 实习岗核心知识点")

    registry = SkillRegistry()
    registry.register(MCP_LEARN_SKILL)
    registry.register(MULTIMODAL_LEARN_SKILL)
    registry.register(CV_LEARN_SKILL)

    print("📚 三个 JD 专项学习 Skill：\n")
    for s in registry.list_all():
        print(f"  🔹 {s.name}: {s.description}")
    print()

    # 场景 1: MCP 协议入门
    print("─── 场景 1: MCP 协议入门 ───")
    result = run_skill(
        MCP_LEARN_SKILL,
        "什么是 MCP 协议？它有哪些核心概念？用初学者能懂的语言解释。",
        config=AgentConfig(verbose=False),
    )
    print(f">>> {result}\n")

    # 场景 2: 多模态模型原理
    print("─── 场景 2: 多模态模型原理 ───")
    result = run_skill(
        MULTIMODAL_LEARN_SKILL,
        "CLIP 模型的核心原理是什么？对比学习是怎么做的？请用简单的数学演示一下。",
        config=AgentConfig(verbose=False),
    )
    print(f">>> {result}\n")

    # 场景 3: CV 基础概念
    print("─── 场景 3: CV 基础概念 ───")
    result = run_skill(
        CV_LEARN_SKILL,
        "什么是 ResNet 的残差连接？为什么它能训练更深的网络？可以用代码演示一下吗？",
        config=AgentConfig(verbose=False),
    )
    print(f">>> {result}\n")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    if not check_config():
        return

    demos = [
        ("纯 LLM vs Agent 对比", demo1_llm_vs_agent),
        ("Agent 多工具组合", demo2_multi_tool),
        ("Skill 驱动 Agent", demo3_single_skill),
        ("编程助手 Skill", demo4_coding_skill),
        ("多 Skill 协作", demo5_multi_skill),
        ("JD 专项 Skill (MCP/多模态/CV)", demo6_jd_skills),
    ]

    print("=" * 70)
    print("  🚀 Agent 从零实现 —— 完整演示")
    print("=" * 70)
    print("\n可用演示:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. 运行全部演示")
    print(f"  0. 退出")
    print()

    choice = input("请选择要运行的演示 (输入数字): ").strip()

    try:
        idx = int(choice)
    except ValueError:
        print("无效输入")
        return

    if idx == 0:
        return
    elif idx == len(demos) + 1:
        for _, func in demos:
            try:
                func()
            except Exception as e:
                print(f"❌ 演示出错: {e}")
                continue
    elif 1 <= idx <= len(demos):
        try:
            demos[idx - 1][1]()
        except Exception as e:
            print(f"❌ 演示出错: {e}")
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
