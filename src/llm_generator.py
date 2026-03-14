from __future__ import annotations

from typing import Any, Dict, List

import anthropic
import google.generativeai as genai

from .utils import get_logger


LOGGER = get_logger(__name__)


PROMPT_TEMPLATE = """You are an AI model explainer. Based on the following information, generate a comprehensive natural language summary report:

Model Type: {model_type}
Task: {task_type}
Training Accuracy: {train_acc}
Test Accuracy: {test_acc}
Top Important Features: {features}
SHAP Analysis: {shap_summary}
LIME Analysis: {lime_summary}

Generate a report with:
1. Model Performance Summary
2. Key Insights from Feature Importance
3. Recommendations for Model Improvement
4. Limitations and Considerations
"""


def _build_prompt(xai_results: Dict[str, Any], model_info: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    shap_top_features: List[str] = xai_results.get("shap", {}).get("top_features", [])
    shap_summary = f"Top features: {shap_top_features}" if shap_top_features else "SHAP summary unavailable"

    lime_summary_items = xai_results.get("lime", {}).get("summary", [])
    lime_summary = ", ".join([f"{k}: {v:.4f}" for k, v in lime_summary_items[:5]]) if lime_summary_items else "LIME summary unavailable"

    return PROMPT_TEMPLATE.format(
        model_type=model_info.get("model_type", "Unknown"),
        task_type=model_info.get("task_type", "Unknown"),
        train_acc=model_info.get("train_acc", "N/A"),
        test_acc=metrics.get("accuracy", metrics.get("rmse", "N/A")),
        features=", ".join(shap_top_features[:5]) if shap_top_features else "N/A",
        shap_summary=shap_summary,
        lime_summary=lime_summary,
    )


def generate_explanation(
    xai_results: Dict[str, Any],
    model_info: Dict[str, Any],
    metrics: Dict[str, Any],
    llm_provider: str,
    api_keys: Dict[str, str],
) -> str:
    """Generate natural-language explanation using Gemini or Claude API."""
    prompt = _build_prompt(xai_results, model_info, metrics)

    if llm_provider.lower() == "gemini":
        api_key = api_keys.get("gemini_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return "Gemini API key not configured. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text or _fallback_report(model_info, metrics, xai_results)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Gemini API failed: %s", e)
            return "LLM call failed. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)

    if llm_provider.lower() == "claude":
        api_key = api_keys.get("claude_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return "Claude API key not configured. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)

        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            if message.content and hasattr(message.content[0], "text"):
                return message.content[0].text
            return _fallback_report(model_info, metrics, xai_results)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Claude API failed: %s", e)
            return "LLM call failed. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)

    return _fallback_report(model_info, metrics, xai_results)


def _fallback_report(model_info: Dict[str, Any], metrics: Dict[str, Any], xai_results: Dict[str, Any]) -> str:
    """Create fallback explanation when external API is unavailable."""
    top_features = xai_results.get("shap", {}).get("top_features", [])
    lines = [
        "## 1. Model Performance Summary",
        f"- Model Type: {model_info.get('model_type', 'Unknown')}",
        f"- Task: {model_info.get('task_type', 'Unknown')}",
        f"- Test Accuracy/Fitness: {metrics.get('accuracy', metrics.get('rmse', 'N/A'))}",
        "",
        "## 2. Key Insights from Feature Importance",
        f"- Top SHAP features: {', '.join(top_features[:5]) if top_features else 'Unavailable'}",
        "",
        "## 3. Recommendations for Model Improvement",
        "- Tune learning rate, batch size, and hidden dimensions further.",
        "- Consider more training data and feature engineering.",
        "",
        "## 4. Limitations and Considerations",
        "- Explanation quality depends on model stability and feature quality.",
        "- LIME local explanations may vary per instance.",
    ]
    return "\n".join(lines)
