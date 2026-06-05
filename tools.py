"""
tools.py —— Tool（工具）系统
=============================
Tool 是 Agent 的最小可调用单元。
每个 Tool 有：名称、描述、参数 schema、执行函数。

描述和参数 schema 是写给 LLM 看的——LLM 靠读这些来决定
"现在该用哪个工具、传什么参数"。
"""

import json
import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo


# ═══════════════════════════════════════════════════════════
# 全局工具注册表
# ═══════════════════════════════════════════════════════════

_registry: list[dict[str, Any]] = []


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, dict],
) -> Callable:
    """
    装饰器：把一个 Python 函数注册为 Agent 可调用的 Tool。

    用法：
        @register_tool(
            name="search_web",
            description="搜索网页获取信息",
            parameters={"query": {"type": "string", "description": "搜索关键词"}}
        )
        def search_web(query: str) -> str:
            ...

    注册后，函数会出现在 get_tools() 返回的列表中，
    并可通过 execute_tool(name, **kwargs) 执行。
    """

    def decorator(func: Callable) -> Callable:
        _registry.append({
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": func,
        })
        return func

    return decorator


def get_tools() -> list[dict[str, Any]]:
    """获取所有已注册的 Tool"""
    return _registry


def get_tool_schemas() -> list[dict]:
    """
    将 Tool 转换为 OpenAI API 的 Function Calling 格式。
    这是喂给 LLM 的标准格式。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": t["parameters"],
                    "required": list(t["parameters"].keys()),
                },
            },
        }
        for t in _registry
    ]


def execute_tool(name: str, **kwargs) -> str:
    """按名称执行一个已注册的 Tool，返回字符串结果"""
    for t in _registry:
        if t["name"] == name:
            try:
                result = t["function"](**kwargs)
                return str(result)
            except Exception as e:
                return f"❌ 工具 '{name}' 执行出错: {e}"
    return f"❌ 错误: 未找到名为 '{name}' 的工具"


def list_tools() -> str:
    """返回所有已注册 Tool 的清单（给调试用）"""
    lines = ["已注册的工具列表:"]
    for t in _registry:
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in t["parameters"].items()
        )
        lines.append(f"  • {t['name']}({params}) - {t['description']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 内置 Tool 定义
# ═══════════════════════════════════════════════════════════

@register_tool(
    name="get_current_time",
    description="获取当前日期和时间。当用户问'今天几号''现在几点'时使用。",
    parameters={
        "timezone": {
            "type": "string",
            "description": "时区，如 Asia/Shanghai、America/New_York，默认 Asia/Shanghai",
        }
    },
)
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Asia/Shanghai")
        timezone = "Asia/Shanghai (fallback, 无法识别指定时区)"
    now = datetime.datetime.now(tz=tz)
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} (时区: {timezone})"


@register_tool(
    name="search_web",
    description="搜索网页获取实时信息。当需要最新资讯、事实查证、数据查询时使用。",
    parameters={
        "query": {"type": "string", "description": "搜索关键词，用中文或英文"}
    },
)
def search_web(query: str) -> str:
    """联网搜索，优先使用 DuckDuckGo，失败时回退到本地知识库。"""
    # ── 尝试真实联网搜索 ──
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"📌 {r['title']}\n   {r['body']}\n   🔗 {r['href']}")

        if results:
            return f"🔍 搜索结果 [{query}]:\n\n" + "\n\n".join(results)
    except Exception:
        pass

    # ── 尝试旧版 duckduckgo_search ──
    try:
        from duckduckgo_search import DDGS as DDGS_old
        results = []
        with DDGS_old() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"📌 {r['title']}\n   {r['body']}\n   🔗 {r['href']}")
        if results:
            return f"🔍 搜索结果 [{query}]:\n\n" + "\n\n".join(results)
    except Exception:
        pass

    # ── 本地知识库（离线回退） ──
    mock_db = {
        "北京天气": "北京今天晴，15°C ~ 28°C，北风3级，空气质量良",
        "上海天气": "上海今天多云转小雨，20°C ~ 26°C，东南风2级",
        "AI新闻": "2026年6月AI行业动态: OpenAI发布GPT-5多模态模型；Anthropic推出Claude 4 Agent框架；谷歌DeepMind开源Gemma 3系列轻量模型",
        "Python": "Python 3.13 已发布，主要特性：无GIL可选模式、改进的错误提示、JIT编译器(实验性)",
        "深度学习": "2026年主流深度学习框架: PyTorch 2.6、TensorFlow 2.18、JAX 0.4。趋势：MLLM多模态模型、Agent框架、边缘AI推理",
        "目标检测": "YOLOv10已发布，主要改进：无NMS训练、Transformer-based检测头、Anchor-free设计。COCO mAP达54.1%",
        "MCP协议": "Model Context Protocol (MCP) 是Anthropic推出的开放标准，用于LLM与外部工具/数据源的安全交互。支持Tools、Resources、Prompts三种原语",
        "CLIP": "CLIP (Contrastive Language-Image Pre-training) 由OpenAI提出，通过对比学习实现图文对齐。使用双塔结构（ViT/ResNet图像编码器 + Transformer文本编码器）",
        "BLIP": "BLIP (Bootstrapping Language-Image Pre-training) 由Salesforce提出，在CLIP基础上增加图像描述生成能力。BLIP-2引入Q-Former桥接视觉和语言",
        "多模态模型": "主流多模态大模型: GPT-4V/4o (OpenAI)、Gemini (Google)、Claude 3.5 (Anthropic)、LLaVA (开源)、Qwen-VL (阿里)。核心技术: 视觉编码器 + 投影层 + LLM解码器",
        "ResNet": "ResNet (残差网络) 由微软提出，核心创新是残差连接(skip connection): 输出=F(x)+x。解决了深层网络的梯度消失问题",
        "Transformer": "Transformer 架构核心: 自注意力(Self-Attention)机制。公式: Attention(Q,K,V)=softmax(QK^T/√d_k)V",
        "图像分割": "主流图像分割模型: SAM (Meta的Segment Anything Model)、Mask R-CNN、U-Net",
        "MCP Tools": "MCP Tools 允许LLM调用外部功能（类似Function Calling）。Server通过 tool/list 列出工具，Client通过 tool/call 执行",
    }

    for key, result in mock_db.items():
        if key in query:
            return f"🔍 搜索结果 [本地-{key}]: {result}"

    return f"🔍 关于 '{query}' 未找到结果。请尝试更具体的关键词。"


@register_tool(
    name="run_python",
    description="在沙箱中执行 Python 代码并返回结果。用于数学计算、数据分析、图表生成。代码中如需输出最终结果，请赋值给变量 'result'（如 result = 3.14 * 2）。",
    parameters={
        "code": {"type": "string", "description": "要执行的 Python 代码"}
    },
)
def run_python(code: str) -> str:
    """安全执行 Python 代码，执行后自动保存到 output/ 目录"""
    import os as _os

    safe_builtins = {
        "__import__": __import__,
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format, "frozenset": frozenset,
        "hex": hex, "int": int, "isinstance": isinstance, "len": len,
        "list": list, "map": map, "max": max, "min": min, "oct": oct,
        "ord": ord, "pow": pow, "range": range, "repr": repr,
        "reversed": reversed, "round": round, "set": set, "slice": slice,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "type": type, "zip": zip, "print": print,
    }
    try:
        # 使用同一个 dict 作为 globals 和 locals，避免类体作用域问题
        # （不同 dict 会导致函数间互相不可见，递归调用 NameError）
        namespace: dict = {"__builtins__": safe_builtins}
        exec(code, namespace, namespace)

        # ─── 自动保存代码到 output/ ───
        _os.makedirs("output", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/code_{ts}.py"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(code)
        saved_note = f"\n💾 代码已自动保存至: {save_path}"
        # ──────────────────────────────

        if "result" in namespace:
            return str(namespace["result"]) + saved_note
        filtered = {k: v for k, v in namespace.items() if not k.startswith("_")}
        if filtered:
            return "\n".join(f"{k} = {v}" for k, v in filtered.items()) + saved_note
        return "代码执行完成（无返回值）" + saved_note
    except Exception as e:
        return f"❌ 执行错误: {type(e).__name__}: {e}"


@register_tool(
    name="read_file",
    description="读取文件内容，返回文本。当需要查看文件内容时使用。",
    parameters={
        "filepath": {"type": "string", "description": "文件路径"}
    },
)
def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"❌ 文件不存在: {filepath}"
    except Exception as e:
        return f"❌ 读取失败: {e}"


@register_tool(
    name="write_file",
    description="将内容写入文件。用于保存代码、报告、数据等到磁盘。重要：生成代码后务必用此工具保存为 .py 文件。",
    parameters={
        "filepath": {"type": "string", "description": "文件路径，如 output/fibonacci.py"},
        "content": {"type": "string", "description": "要写入的文件内容"},
    },
)
def write_file(filepath: str, content: str) -> str:
    import os as _os
    try:
        dirpath = _os.path.dirname(filepath)
        if dirpath and not _os.path.exists(dirpath):
            _os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ 文件已保存: {filepath} ({len(content)} 字符)"
    except Exception as e:
        return f"❌ 写入失败: {e}"


# ═══════════════════════════════════════════════════════════
# 初始化时打印可用工具
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(list_tools())
    print()

    # 手动测试工具
    print("--- 测试 search_web ---")
    print(execute_tool("search_web", query="深圳天气"))
    print()

    print("--- 测试 run_python ---")
    print(execute_tool("run_python", code="result = sum(range(1, 101))"))
    print()

    print("--- 测试 get_current_time ---")
    print(execute_tool("get_current_time"))
