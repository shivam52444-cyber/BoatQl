"""
The structured output schema for chart selection. Same pattern as
SQLGenerationResult: the LLM fills in this schema (via tool-calling),
it never writes or executes actual chart code.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "pie", "number_card", "table"] = Field(
        description="The type of chart that best fits this data's shape."
    )
    x_column: Optional[str] = Field(
        default=None,
        description="Column to use for the x-axis / categories / labels. Required for bar, line, pie. Omit for number_card and table."
    )
    y_column: Optional[str] = Field(
        default=None,
        description="Column to use for the y-axis / values. Required for bar, line, number_card. Omit for pie (uses x_column for labels and assumes a single numeric value column) and table."
    )
    title: str = Field(description="A short, human-readable title for the chart.")
    reasoning: str = Field(description="One sentence on why this chart type fits this data shape.")