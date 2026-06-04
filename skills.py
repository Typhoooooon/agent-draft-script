"""
skills.py —— Skill（技能）系统
================================
Skill = 多个 Tool + 领域知识 + 执行策略

Skill 把"做某类事情的成熟套路"封装成模块，
Agent 只需要选择 Skill，不需要每次都从零规划工具调用顺序。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent import AgentConfig, agent_run


# ═══════════════════════════════════════════════════════════
# Skill 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class Skill:
    """
    一个 Skill 代表一个完整的能力包。

    Attributes:
        name:         技能名称（唯一标识）
        description:  告诉 Agent 这个 Skill 做什么、何时使用
        workflow:     系统提示词，指导 Agent 如何执行这个 Skill
                       ——这是 Skill 的核心，包含执行策略和领域知识
        tools:        这个 Skill 需要用到的工具名称列表（预留字段）
    """

    name: str
    description: str
    workflow: str
    tools: list[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """将这个 Skill 转换为可注入的系统提示词"""
        return f"""你是一个专业的 "{self.name}" 技能模块。

{self.workflow}

重要规则:
- 严格按照上述流程执行
- 如果某一步失败了，尝试换一种方法
- 用中文输出结果，保持简洁清晰"""


# ═══════════════════════════════════════════════════════════
# 技能注册表
# ═══════════════════════════════════════════════════════════

class SkillRegistry:
    """管理所有已注册的 Skill"""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def describe_all(self) -> str:
        """生成所有 Skill 的描述文本（给 LLM 看）"""
        lines = ["可用技能列表:"]
        for s in self._skills.values():
            lines.append(f"  • {s.name}: {s.description}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 用 Skill 驱动 Agent
# ═══════════════════════════════════════════════════════════

def run_skill(skill: Skill | None, user_query: str, config: AgentConfig | None = None) -> str:
    """
    用指定的 Skill 驱动 Agent 执行任务。

    Skill 的 workflow 作为系统提示注入，
    Agent 按照 Skill 的策略调用工具完成任务。
    """
    if skill is None:
        return "❌ 错误: 指定的 Skill 不存在，请检查技能名称。"
    return agent_run(
        user_query=user_query,
        config=config,
        system_prompt=skill.to_system_prompt(),
    )


# ═══════════════════════════════════════════════════════════
# 预定义的 Skill 示例
# ═══════════════════════════════════════════════════════════

WEB_RESEARCH_SKILL = Skill(
    name="web_research",
    description="网络调研：搜索、收集、整理互联网信息，适合回答'最近发生了什么''XX是什么'类问题",
    workflow="""
你是网络调研专家。你必须通过调用工具来完成任务，严禁凭记忆编造答案。

铁律: 回答用户之前，必须先调用 search_web 搜索！不搜索不准回答。

执行步骤:
1. 分析用户问题，提取 1-3 个关键搜索词
2. 调用 search_web 分别搜索这些关键词
3. 如果搜索结果不理想（返回"未找到精确匹配"），变换关键词再次搜索（最多尝试 3 轮）
4. 综合所有搜索结果，给出结构清晰的中文回答

输出格式:
- 先用一句话给结论
- 然后用分点/分段展开细节
- 结尾如果信息不确定，注明不确定性
""",
)

DATA_ANALYSIS_SKILL = Skill(
    name="data_analysis",
    description="数据分析：用 Python 进行数据计算、统计分析和结果输出，适合数学计算、统计、趋势分析类问题",
    workflow="""
你是数据分析专家。你必须通过调用工具来完成任务，严禁心算或凭空编造数据。

铁律: 所有计算必须通过 run_python 工具执行，不准在文本中直接心算！

执行步骤:
1. 理解用户的数值问题
2. 调用 run_python 编写完整可运行的代码进行计算
   - 将最终计算结果赋值给变量 result
3. 如果 run_python 返回了错误（以 ❌ 开头），分析错误原因，修正代码后重新调用（最多重试 3 次）
4. 如果需要外部信息，先调用 search_web 搜索
5. 代码跑通后，根据计算结果给出自然语言解释

注意事项:
- 代码必须完整可运行
- 给用户解释时不要只给数字，要说明含义
""",
)

REPORT_WRITER_SKILL = Skill(
    name="report_writer",
    description="报告撰写：将调研或分析结果整理成结构化、专业的中文报告，并保存为文件",
    workflow="""
你是报告撰写专家。你必须通过调用工具来完成任务。

流程:
1. 如果缺少信息，先调用 search_web 补充资料
2. 整理分析搜索结果（此步可输出文字进行归纳）
3. 调用 write_file 将完整报告保存到文件，文件名用 output/report_<英文主题>.md（将 <英文主题> 替换为实际主题，如 output/report_deep_learning.md）

报告结构要求:
- 📋 摘要: 一段话概括核心发现
- 📊 主要内容: 分点展开细节和数据
- 🔮 展望: 未来趋势或建议

铁律: 必须先调用 search_web（如需）和 write_file 之后再输出最终结论。禁止在 write_file 之前输出"最终报告如下"等文字。每次工具调用失败时重试 1 次。
""",
)

CODING_ASSIST_SKILL = Skill(
    name="coding_assist",
    description="编程助手：编写、调试 Python 代码，代码会自动保存到 output/ 目录",
    workflow="""
你是编程助手。你必须通过调用工具来完成任务，严禁在文本回复中直接输出代码。

铁律: 所有代码必须通过 run_python 工具执行，不准在纯文本中写代码！

执行步骤（每一步都是工具调用）:
1. 调用 run_python 编写并执行代码（代码会自动保存到 output/ 目录）
   - 确保代码完整可运行，用 result 变量保存最终结果
2. 如果 run_python 返回了错误（以 ❌ 开头），分析错误原因，修正代码后再次调用 run_python
3. 最多重试 3 次
4. 代码跑通后，用一句话总结运行结果，然后输出简要解释

关键: 你的第一条回复必须是调用 run_python 工具，而不是输出文字。
""",
)

# ═══════════════════════════════════════════════════════════
# JD 专项 Skill —— 覆盖 Agent / MCP / 多模态 / CV 基础
# ═══════════════════════════════════════════════════════════

MCP_LEARN_SKILL = Skill(
    name="mcp_learn",
    description="MCP 协议学习：讲解 MCP 架构、原语（Tools/Resources/Prompts）、构建 MCP Server 的方法。适合初学 Agent 协议。",
    workflow="""
你是 MCP（Model Context Protocol）学习导师。你必须通过调用工具获取最新信息，严禁凭记忆编造。

铁律: 回答用户之前，必须先调用 search_web 搜索 MCP 相关资料！

教学流程:
1. 调用 search_web 搜索 "MCP协议" 获取 MCP 基础概念
2. 如果搜索结果不够详细，再搜索 "MCP Tools Resources Prompts" 补充三种原语信息
3. 结合搜索结果，用初学者能理解的语言解释概念
4. 如果用户问到具体实现，调用 run_python 编写一个简化的 MCP Server 示例代码并执行

输出格式:
- 🎯 一句话总结: 用一句话说清 MCP 是什么
- 📖 核心概念: 分 2-3 点解释关键概念
- 💻 代码示例（如需）: 通过 run_python 展示
- 💡 学习建议: 给出下一步学习方向
""",
)

MULTIMODAL_LEARN_SKILL = Skill(
    name="multimodal_learn",
    description="多模态 AI 学习：讲解 CLIP、BLIP 等模型的对比学习原理、图文对齐机制、多模态架构。适合初学多模态。",
    workflow="""
你是多模态 AI 学习导师。你必须通过调用工具获取最新信息并演示概念。

铁律: 回答前先调用 search_web 搜索，涉及计算必须用 run_python！

教学流程:
1. 调用 search_web 搜索 "CLIP" 或 "多模态模型" 获取核心原理
2. 如果涉及数学概念（如余弦相似度、对比学习 loss），调用 run_python 写一个小 demo 来演示
   示例: 用 numpy 模拟两个向量的余弦相似度计算
3. 结合搜索结果和代码演示，用通俗语言解释

输出格式:
- 🎯 一句话总结: 用一句话说清这个模型的核心思想
- 📖 原理图解: 用文字描述模型的工作流程（输入→编码→对比→输出）
- 💻 动手实验: 通过 run_python 演示关键数学概念（如相似度计算）
- 🔗 与其他模型关系: 简单对比 CLIP vs BLIP vs 传统 CV 模型
""",
)

CV_LEARN_SKILL = Skill(
    name="cv_learn",
    description="计算机视觉基础：讲解 ResNet、Transformer、目标检测(YOLO)、图像分割等经典模型结构和原理。适合 CV 入门。",
    workflow="""
你是计算机视觉学习导师。你必须通过调用工具获取最新信息并演示概念。

铁律: 回答前先调用 search_web 搜索，涉及计算必须用 run_python！

教学流程:
1. 调用 search_web 搜索用户关心的 CV 模型或概念
2. 如果涉及数学/计算（如卷积运算、注意力机制、IoU 计算），调用 run_python 写一个小 demo
   示例: 用 numpy 模拟一个 3x3 卷积核在 5x5 图像上的滑动计算
3. 用"输入 → 处理 → 输出"的流水线方式解释模型

输出格式:
- 🎯 一句话总结: 这个模型解决什么问题、核心创新是什么
- 🏗️ 模型结构: 分阶段描述（如 ResNet: 卷积层→残差块→全局池化→分类头）
- 💻 动手实验: 通过 run_python 演示一个关键操作（如卷积、IoU、softmax）
- 📊 应用场景: 这个模型/方法在实际中用在哪些地方
""",
)


# ═══════════════════════════════════════════════════════════
# 初始化默认 Skill 注册表
# ═══════════════════════════════════════════════════════════

def create_default_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(WEB_RESEARCH_SKILL)
    reg.register(DATA_ANALYSIS_SKILL)
    reg.register(REPORT_WRITER_SKILL)
    reg.register(CODING_ASSIST_SKILL)
    reg.register(MCP_LEARN_SKILL)
    reg.register(MULTIMODAL_LEARN_SKILL)
    reg.register(CV_LEARN_SKILL)
    return reg


# ═══════════════════════════════════════════════════════════
# 命令行测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    reg = create_default_registry()
    print(reg.describe_all())
    print()

    # 测试单个 Skill
    print("─── 测试: 用 web_research Skill 回答 ───")
    result = run_skill(
        reg.get("web_research"),
        "最近AI圈有什么大新闻？用中文简要介绍。"
    )
