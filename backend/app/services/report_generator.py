import json
import logging
import os
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def generate_groq_report(structured_result: Dict[str, Any]) -> Dict[str, str]:
    """Generate a readable agronomic summary using Groq if enabled,

    otherwise fall back to knowledge base facts safely.
    """
    enable_groq = os.getenv("ENABLE_GROQ_REPORTS", "false").strip().lower() in ("true", "1", "yes")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

    # Fallback template from structured knowledge-base facts
    pred = structured_result.get("prediction", {})
    disease = pred.get("disease", "Unknown")
    confidence = pred.get("confidence_percent", 0.0)
    mgmt = structured_result.get("management", {})
    immediate = mgmt.get("immediate_actions", [])
    preventive = mgmt.get("preventive_measures", [])

    fallback_summary = (
        f"Diagnostic Screening: Identified {disease} with {confidence:.1f}% confidence. "
        f"Immediate action recommended: {immediate[0] if immediate else 'Consult an extension agent.'} "
        f"Preventive strategy: {preventive[0] if preventive else 'Maintain good orchard sanitation.'}"
    )

    if not enable_groq or not api_key:
        return {
            "generated_by": "knowledge_base",
            "text": fallback_summary
        }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        # Exclude bulky binary/base64 data from prompt
        clean_payload = {
            "fruit": structured_result.get("fruit"),
            "prediction": structured_result.get("prediction"),
            "environment": structured_result.get("environment"),
            "management": structured_result.get("management"),
            "explanation_method": structured_result.get("explanation", {}).get("method")
        }
        payload_str = json.dumps(clean_payload, ensure_ascii=False)

        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=300,
            timeout=8.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain a plant-disease screening result to a grower. "
                        "Use ONLY the supplied prediction, explanation, environment, and treatment data. "
                        "Do not change the predicted disease, invent sources, or invent chemical products, "
                        "dosages, mixing ratios, application schedules, or safety intervals. "
                        "Keep the summary to 2-3 clear, practical sentences and clearly distinguish model "
                        "confidence from clinical certainty."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Write a concise plain-language report from this JSON only:\n{payload_str}",
                },
            ],
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return {
                "generated_by": "groq",
                "text": content.strip()
            }
    except Exception as exc:
        logger.warning("Groq report generation unavailable: %s. Falling back to knowledge base.", exc)

    return {
        "generated_by": "knowledge_base_fallback",
        "text": fallback_summary
    }

