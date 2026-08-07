"""
Real chart rendering. The LLM never writes or executes any of this code --
it only returns a ChartSpec (chart_type + column choices), and this module
dispatches to the matching pre-written, trusted function.

Each function takes rows (list of dicts) + a ChartSpec and returns a dict
with the rendered chart as a base64 PNG (or, for number_card/table, plain
structured data -- no image needed for those).
"""

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")  # no display backend needed, we're rendering to bytes
import matplotlib.pyplot as plt

from chart_spec import ChartSpec


class ChartRenderError(Exception):
    """Raised when a ChartSpec can't actually be rendered against the given data."""
    pass


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _validate_columns(rows: list[dict], *columns: str | None):
    if not rows:
        raise ChartRenderError("No rows to chart.")
    available = set(rows[0].keys())
    for col in columns:
        if col is not None and col not in available:
            raise ChartRenderError(f"Column '{col}' not found in result columns: {sorted(available)}")


def render_bar(rows: list[dict], spec: ChartSpec) -> dict:
    _validate_columns(rows, spec.x_column, spec.y_column)
    if not spec.x_column or not spec.y_column:
        raise ChartRenderError("Bar chart requires both x_column and y_column.")

    labels = [str(r[spec.x_column]) for r in rows]
    values = [r[spec.y_column] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_xlabel(spec.x_column)
    ax.set_ylabel(spec.y_column)
    ax.set_title(spec.title)
    plt.xticks(rotation=45, ha="right")

    return {"type": "image", "chart_type": "bar", "image_base64": _fig_to_base64(fig)}


def render_line(rows: list[dict], spec: ChartSpec) -> dict:
    _validate_columns(rows, spec.x_column, spec.y_column)
    if not spec.x_column or not spec.y_column:
        raise ChartRenderError("Line chart requires both x_column and y_column.")

    labels = [str(r[spec.x_column]) for r in rows]
    values = [r[spec.y_column] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, values, marker="o", color="#4C72B0")
    ax.set_xlabel(spec.x_column)
    ax.set_ylabel(spec.y_column)
    ax.set_title(spec.title)
    plt.xticks(rotation=45, ha="right")

    return {"type": "image", "chart_type": "line", "image_base64": _fig_to_base64(fig)}


def render_pie(rows: list[dict], spec: ChartSpec) -> dict:
    _validate_columns(rows, spec.x_column, spec.y_column)
    if not spec.x_column:
        raise ChartRenderError("Pie chart requires x_column (labels).")

    value_col = spec.y_column
    if value_col is None:
        numeric_cols = [k for k, v in rows[0].items() if k != spec.x_column and isinstance(v, (int, float))]
        if not numeric_cols:
            raise ChartRenderError("Pie chart needs a numeric value column, none found.")
        value_col = numeric_cols[0]

    labels = [str(r[spec.x_column]) for r in rows]
    values = [r[value_col] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%")
    ax.set_title(spec.title)

    return {"type": "image", "chart_type": "pie", "image_base64": _fig_to_base64(fig)}


def render_number_card(rows: list[dict], spec: ChartSpec) -> dict:
    if not rows:
        raise ChartRenderError("No rows to render as a number card.")

    if spec.y_column:
        _validate_columns(rows, spec.y_column)
        value = rows[0][spec.y_column]
    else:
        # fall back to the first numeric value in the first row
        numeric_vals = [v for v in rows[0].values() if isinstance(v, (int, float))]
        if not numeric_vals:
            raise ChartRenderError("No numeric column found for number_card.")
        value = numeric_vals[0]

    return {"type": "number_card", "chart_type": "number_card", "title": spec.title, "value": value}


def render_table(rows: list[dict], spec: ChartSpec) -> dict:
    if not rows:
        raise ChartRenderError("No rows to render as a table.")
    return {"type": "table", "chart_type": "table", "title": spec.title, "columns": list(rows[0].keys()), "rows": rows}


_RENDERERS = {
    "bar": render_bar,
    "line": render_line,
    "pie": render_pie,
    "number_card": render_number_card,
    "table": render_table,
}


def render_chart(rows: list[dict], spec: ChartSpec) -> dict[str, Any]:
    """Dispatches to the correct trusted renderer based on spec.chart_type.
    Raises ChartRenderError if the spec doesn't match the data (e.g. LLM
    picked a column that doesn't exist in the result set)."""
    renderer = _RENDERERS.get(spec.chart_type)
    if renderer is None:
        raise ChartRenderError(f"Unknown chart_type: {spec.chart_type}")
    return renderer(rows, spec)