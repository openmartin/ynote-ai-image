import logging
import os
from enum import Enum
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

llm_api_key = os.getenv("QWEN_API_KEY")
llm_api_url = os.getenv("QWEN_API_URL")
llm_model_name = os.getenv("QWEN_MODE_NAME")

llm = init_chat_model(model=llm_model_name, model_provider="openai",
                      base_url=llm_api_url,
                      api_key=llm_api_key,
                      temperature=0.5)


class SvgType(str, Enum):
    TREELIKE = "treelike"
    PARALLEL = "parallel"


class SubThemeItem(BaseModel):
    emoji: str = Field(description="a single, highly relevant emoji that best symbolizes the concept")
    title: str = Field(description="a clear and concise title (a few words)")
    description: str = Field(description="an insightful description under 50 words")


class Outline(BaseModel):
    svgStyle: SvgType = Field(description="svg style treelike or parallel")
    theme: str = Field(description="main theme of the text")
    subThemeList: List[SubThemeItem] = Field(description="3 to 8 key sub-themes or main points")

    class Config:
        json_encoders = {
            SvgType: lambda v: v.value
        }


PROMPT = """Act as an expert summarization and information structuring assistant. Your task is to analyze the provided text and generate a concise, structured summary in a specific JSON format.

Process the text by:

Identifying the core, overarching theme.

Extracting the key sub-themes or main points. The number of sub-themes is determined by their logical relationship (see svgStyle rule below).

For each sub-theme:

Assign a single, highly relevant emoji that best symbolizes the concept.

Write a clear and concise title under 20 words.

Write an insightful description under 50 words.

Determine the Structure and Count:

Choose "svgStyle": "treelike" if the sub-themes are hierarchical, building upon a core idea, or following a logical progression. For this style, extract 3 to 5 sub-themes.

Choose "svgStyle": "parallel" if the sub-themes are more than one subject, each sub-theme for one subject. For this style, extract 3 to 8 sub-themes. For this style, add subject name to sub-themes title.

Output the summary exclusively as a valid JSON object following this exact structure without any additional commentary, explanations, or markdown formatting:

Always use the language specified by the locale = **zh**.

{{
    "svgStyle": "treelike | parallel",
    "theme": "The Central Theme",
    "mainContent": [
    {{
        "emoji": "🚀",
        "title": "Sub-theme Title",
        "description": "A concise description of this point in 50 words or less."
    }},
// ... more sub-theme objects
]
}}

Text to Summarize:
{content}
"""

def summarize_to_json(text: str):
    structured_llm = llm.with_structured_output(Outline)
    response = structured_llm.invoke([HumanMessage(content=PROMPT.format(content=text))], temperature=0.5)
    logger.info(response)
    return response

