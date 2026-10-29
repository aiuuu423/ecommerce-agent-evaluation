from app.prompts.base import PromptSnapshot, make_prompt_snapshot

V1_PROMPT_NAME = "real-llm-v1"
V1_PROMPT_VERSION = "1.0.0"
V1_CHANGE_NOTE = "Initial basic business-analysis agent prompt."
V1_PROMPT_TEXT = """你是一名电商经营分析助手，基于 Synthetic E-commerce Data 回答经营分析问题。

可用工具：
- 你可以调用工具查询商品、销售、流量、投放数据，并基于已取得的工具结果计算指标。
- 需要数据时优先调用工具获取事实，不要编造工具结果。

工作方式：
- 需要数据时调用工具；取得足够事实后直接给出回答。
- 回答应包含结论和支撑结论的关键数据。
- 如果问题无法由工具数据支持，请明确说明这一限制。
"""


def v1_prompt_snapshot() -> PromptSnapshot:
    return make_prompt_snapshot(
        name=V1_PROMPT_NAME,
        version=V1_PROMPT_VERSION,
        text=V1_PROMPT_TEXT,
        change_note=V1_CHANGE_NOTE,
        source_file="app/prompts/v1.py",
    )
