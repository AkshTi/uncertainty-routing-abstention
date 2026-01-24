"""
Unified Prompt System - PUBLICATION READY
Addresses ChatGPT feedback: "Unify prompts (ALL templates/domains): enforce one-line mutually exclusive output"

This module ensures ALL experiments use the SAME prompt format to eliminate confounding variables.
"""

def unified_prompt(question: str) -> str:
    """
    UNIFIED prompt format for ALL experiments.

    CRITICAL: Do NOT modify this prompt format without updating ALL experiments.
    Any changes here affect baseline comparisons across all results.

    Design rationale:
    - One-line output only (prevents "answer + UNCERTAIN" mixing)
    - Mutually exclusive: answer OR UNCERTAIN, not both
    - No system prompt variations (eliminates prompt engineering confounds)
    - Simple format works across all domains

    Args:
        question: The question to ask

    Returns:
        Formatted prompt string
    """
    return f"""Question: {question}

Answer with EXACTLY one line:
- Your answer (one line) OR
- Exactly: UNCERTAIN

No explanation. No extra text.

Answer:"""


def unified_prompt_strict(question: str) -> str:
    """
    STRICTEST version - for final publication experiments.
    Even more explicit about format requirements.
    """
    return f"""Question: {question}

IMPORTANT: Return EXACTLY one line.
- If you know the answer: write it in one line
- If you don't know: write exactly "UNCERTAIN"

Answer:"""


def unified_prompt_minimal(question: str) -> str:
    """
    Minimal version - for comparison/ablation studies.
    Simplest possible prompt to test pure steering effects.
    """
    return f"Question: {question}\nAnswer:"


# Default: use the main unified prompt
format_prompt = unified_prompt


# ============================================================================
# Template Validation
# ============================================================================

def validate_response_format(response: str) -> dict:
    """
    Validate that a response follows the expected one-line format.

    Returns:
        dict with:
            - is_valid: bool
            - issue: str (description of problem if invalid)
            - line_count: int
    """
    lines = [l.strip() for l in response.split('\n') if l.strip()]

    if len(lines) == 0:
        return {"is_valid": False, "issue": "Empty response", "line_count": 0}

    if len(lines) == 1:
        return {"is_valid": True, "issue": None, "line_count": 1}

    # Multiple lines - check if it's answer + explanation
    if lines[0].upper() == "UNCERTAIN" and len(lines) > 1:
        return {
            "is_valid": False,
            "issue": "UNCERTAIN with explanation (should be one line only)",
            "line_count": len(lines)
        }

    return {
        "is_valid": False,
        "issue": f"Multi-line response ({len(lines)} lines)",
        "line_count": len(lines)
    }


# ============================================================================
# Prompt Variants (for ablation studies only)
# ============================================================================

ABLATION_PROMPTS = {
    "unified": unified_prompt,
    "strict": unified_prompt_strict,
    "minimal": unified_prompt_minimal,
}


def get_prompt_variant(variant: str = "unified") -> callable:
    """
    Get a specific prompt variant for ablation studies.

    WARNING: Only use this for controlled ablation experiments.
    For all main experiments, use unified_prompt() directly.
    """
    if variant not in ABLATION_PROMPTS:
        raise ValueError(f"Unknown variant '{variant}'. Use one of: {list(ABLATION_PROMPTS.keys())}")
    return ABLATION_PROMPTS[variant]


# ============================================================================
# Migration Helper
# ============================================================================

def old_format_prompt(question: str, system_prompt: str = "neutral", context=None) -> str:
    """
    OLD format from data_preparation.py - DEPRECATED

    This is kept for backwards compatibility only.
    DO NOT USE in new code - use unified_prompt() instead.
    """
    import warnings
    warnings.warn(
        "old_format_prompt is deprecated. Use unified_prompt() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Map old system_prompt to new unified format
    return unified_prompt(question)
