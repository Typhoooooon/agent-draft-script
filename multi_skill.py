"""
multi_skill.py —— 多 Skill 协作编排器
======================================
这是实习岗位核心能力的直接体现：
让 Agent 根据任务自动选择多个 Skill 协作完成复杂任务。

架构:
    User Query
        ↓
    Orchestrator (LLM 决策)
        ↓
    ┌─────┬─────┬─────┐
    Skill1 Skill2 Skill3   ← LLM 决定顺序和依赖
    └─────┴─────┴─────┘
        ↓
    最终汇总输出
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from agent import AgentConfig, agent_run
from skills import (
    Skill,
    SkillRegistry,
    create_default_registry,
)

load_dotenv()

# 延迟初始化，避免 import 时就需要 API key
_mclient: OpenAI | None = None


def _get_client() -> OpenAI:
    global _mclient
    if _mclient is None:
        _mclient = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
    return _mclient


# ═══════════════════════════════════════════════════════════
# 多 Skill 编排器
# ═══════════════════════════════════════════════════════════

@dataclass
class OrchestratorConfig:
    """编排器配置"""
    model: str = "deepseek-chat"
    max_skills: int = 5      # 最多使用几个 Skill
    verbose: bool = True


class MultiSkillOrchestrator:
    """
    多 Skill 编排器。

    工作流程:
    1. 列出所有可用 Skill 给 LLM
    2. LLM 决定需要哪些 Skill、按什么顺序执行
    3. 逐个执行 Skill，结果传给下一个 Skill 做上下文
    4. 最后 LLM 汇总所有 Skill 的输出，生成最终答案
    """

    def __init__(self, registry: SkillRegistry, config: OrchestratorConfig | None = None):
        self.registry = registry
        self.config = config or OrchestratorConfig()

    def _plan_skills(self, user_query: str) -> list[str]:
        """让 LLM 根据用户问题决定使用哪些 Skill"""
        skills_desc = self.registry.describe_all()
        available_names = [s.name for s in self.registry.list_all()]

        prompt = f"""你是一个任务规划器。根据用户的问题，从可用技能中选择最合适的技能来完成该任务。

{skills_desc}

要求:
- 最多选择 {self.config.max_skills} 个技能
- 按执行顺序排列
- 技能名称必须从上述列表中选择，不要编造

请以 JSON 格式返回:
{{"skills": ["skill_name1", "skill_name2"], "reason": "选择理由"}}

用户问题: {user_query}"""

        resp = _get_client().chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        try:
            plan = json.loads(resp.choices[0].message.content or "{}")
            raw_skills = plan.get("skills", [])
            # 过滤掉不存在的技能名
            valid_skills = [s for s in raw_skills if s in available_names]
            if len(valid_skills) < len(raw_skills):
                print(f"⚠️  规划了无效技能，已过滤: {set(raw_skills) - set(valid_skills)}")
            return valid_skills[:self.config.max_skills]
        except json.JSONDecodeError as e:
            print(f"⚠️  技能规划 JSON 解析失败: {e}")
            return []

    def execute(self, user_query: str) -> dict[str, str]:
        """
        执行多 Skill 协作。

        Returns:
            {
                "plan": ["skill1", "skill2"],
                "results": {"skill1": "结果1", "skill2": "结果2"},
                "final_answer": "最终汇总答案"
            }
        """
        log = print if self.config.verbose else lambda *a, **kw: None

        log("=" * 60)
        log("🎯 多 Skill 编排器启动")
        log(f"📝 用户问题: {user_query}")
        log(f"📦 可用 Skill: {[s.name for s in self.registry.list_all()]}")
        log("=" * 60)

        # Step 1: 规划
        log("\n🧠 规划阶段: LLM 分析任务，选择 Skill...")
        skill_names = self._plan_skills(user_query)

        if not skill_names:
            log("⚠️ 未能规划出合适的 Skill 组合，使用单个通用 Agent")
            return {
                "plan": [],
                "results": {},
                "final_answer": agent_run(user_query),
            }

        log(f"📋 执行计划: {' → '.join(skill_names)}")

        # Step 2: 逐个执行 Skill
        results: dict[str, str] = {}
        accumulated_context = ""  # 累积上下文，传给后续 Skill

        for i, skill_name in enumerate(skill_names):
            skill = self.registry.get(skill_name)
            if not skill:
                log(f"⚠️  Skill '{skill_name}' 未找到，跳过")
                continue

            log(f"\n{'─' * 60}")
            log(f"🔵 执行 Skill [{i+1}/{len(skill_names)}]: {skill_name}")

            # 构造带上下文的查询
            if accumulated_context:
                enriched_query = f"""之前的执行结果:
{accumulated_context}

当前任务: {user_query}

请基于之前的执行结果，完成你负责的部分。"""
            else:
                enriched_query = user_query

            # 运行 Skill
            result = agent_run(
                user_query=enriched_query,
                config=AgentConfig(verbose=self.config.verbose),
                system_prompt=skill.to_system_prompt(),
            )

            results[skill_name] = result
            # 截断过长的结果，防止上下文溢出（保留首尾各 400 字符）
            if len(result) > 1000:
                truncated = result[:400] + f"\n... (中间省略 {len(result) - 800} 字符) ...\n" + result[-400:]
            else:
                truncated = result
            accumulated_context += f"\n\n[{skill_name} 输出]:\n{truncated}"

        # Step 3: 汇总
        log(f"\n{'─' * 60}")
        log("🔷 汇总阶段: 整合所有 Skill 输出...")

        summary_prompt = f"""你是一个任务汇总器。以下是多个技能模块的执行结果，请将其整合为一份完整、连贯的回答。

用户原始问题: {user_query}

各模块输出:
{accumulated_context}

请整合为一份完整的答案，保持结构清晰、逻辑连贯。直接输出最终答案，不要加"好的我来汇总"之类的开场白。"""

        resp = _get_client().chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        final_answer = resp.choices[0].message.content or ""

        log(f"\n{'=' * 60}")
        log("✅ 多 Skill 协作完成")
        log(f"{'=' * 60}")

        return {
            "plan": skill_names,
            "results": results,
            "final_answer": final_answer,
        }


# ═══════════════════════════════════════════════════════════
# 命令行测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    registry = create_default_registry()
    orchestrator = MultiSkillOrchestrator(registry)

    # 一个需要多 Skill 协作的复杂任务
    query = "调研最新的深度学习框架发展趋势，分析对比它们的优劣势，最后生成一份简短的技术报告。"

    print(f"📝 测试查询: {query}\n")
    output = orchestrator.execute(query)

    print("\n" + "=" * 60)
    print("🎯 最终答案:")
    print("=" * 60)
    print(output["final_answer"])
