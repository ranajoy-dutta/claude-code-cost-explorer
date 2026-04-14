"""Cost calculation for Claude API usage based on token counts."""

PRICING = [
    ("claude-opus-4",   {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50}),
    ("claude-sonnet-4", {"input": 3.00,  "output": 15.00, "cache_write": 3.75,  "cache_read": 0.30}),
    ("claude-haiku-4",  {"input": 0.80,  "output": 4.00,  "cache_write": 1.00,  "cache_read": 0.08}),
]
DEFAULT_RATES = {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30}


def get_rates(model: str) -> dict:
    model_lower = (model or "").lower()
    for prefix, rates in PRICING:
        if model_lower == prefix or model_lower.startswith(prefix + "-"):
            return dict(rates)
    return dict(DEFAULT_RATES)


def calculate_cost(model: str, usage: dict) -> float:
    if not usage:
        return 0.0
    rates = get_rates(model)
    m = 1_000_000
    return (
        usage.get("input_tokens", 0) / m * rates["input"]
        + usage.get("output_tokens", 0) / m * rates["output"]
        + usage.get("cache_creation_input_tokens", 0) / m * rates["cache_write"]
        + usage.get("cache_read_input_tokens", 0) / m * rates["cache_read"]
    )
