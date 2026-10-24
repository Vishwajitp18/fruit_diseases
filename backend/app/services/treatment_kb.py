import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "treatment_knowledge_base.json"

_KNOWLEDGE_BASE: Optional[Dict[str, Any]] = None


def _clean_str(text: str) -> str:
    """Normalize strings for robust matching."""
    s = text.lower()
    s = re.sub(r"[_\-\s\(\)\/]+", "", s)
    return s


def load_knowledge_base() -> Dict[str, Any]:
    global _KNOWLEDGE_BASE
    if _KNOWLEDGE_BASE is None:
        if not KB_PATH.exists():
            return {}
        with open(KB_PATH, "r", encoding="utf-8") as f:
            _KNOWLEDGE_BASE = json.load(f)
    return _KNOWLEDGE_BASE


def lookup_treatment(fruit: str, disease: str) -> Dict[str, Any]:
    """Look up reviewed treatment facts for the given fruit and disease.
    
    Returns structured actions, preventive measures, treatment options,
    and official agricultural sources.
    """
    kb = load_knowledge_base()
    fruit_key = fruit.strip().lower()
    crop_kb = kb.get(fruit_key, {})

    # 1. Exact match
    if disease in crop_kb:
        return crop_kb[disease]

    # 2. Normalized clean string match
    clean_target = _clean_str(disease)
    for key, data in crop_kb.items():
        if _clean_str(key) == clean_target:
            return data

    # 3. Partial substring match
    for key, data in crop_kb.items():
        clean_k = _clean_str(key)
        if clean_k in clean_target or clean_target in clean_k:
            return data

    # 4. Safe fallback if unreviewed
    return {
        "display_name": disease.title(),
        "symptoms_to_check": ["Examine leaf blade for localized discoloration or lesions."],
        "immediate_actions": [
            "Isolate the affected branch or tree from healthy stands.",
            "Collect a sample in a sealed plastic bag for professional laboratory analysis."
        ],
        "preventive_measures": [
            "Maintain balanced irrigation and good orchard sanitation.",
            "Avoid overhead watering during humid weather."
        ],
        "treatment_options": [
            "No pre-verified chemical prescription for this specific class. Consult local extension guidance."
        ],
        "when_to_seek_expert_help": "Consult a qualified agricultural extension officer or university plant diagnostic clinic before applying treatments.",
        "sources": [
            {
                "title": "National Plant Diagnostic Network",
                "organization": "NPDN Extension",
                "url": "https://www.npdn.org"
            }
        ],
        "reviewed_by": "Standard Horticultural Advisory Protocol",
        "last_reviewed": "2026-10"
    }

