"""Cost calculation for Claude API usage based on token counts."""

# Rates per million tokens. More specific prefixes MUST come first — lookup
# picks the first entry where model == prefix or model.startswith(prefix + "-").
# Source: https://platform.claude.com/docs/en/about-claude/pricing
PRICING = [
    # Opus 4.5+ dropped to 1/3 of the Opus 4/4.1 rate
    (
        "claude-opus-4-7",
        {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    ),
    (
        "claude-opus-4-6",
        {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    ),
    (
        "claude-opus-4-5",
        {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    ),
    (
        "claude-opus-4-1",
        {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    ),
    (
        "claude-opus-4",
        {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    ),
    (
        "claude-sonnet-4",
        {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    ),
    (
        "claude-haiku-4-5",
        {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    ),
    (
        "claude-haiku-3-5",
        {"input": 0.80, "output": 4.00, "cache_write": 1.00, "cache_read": 0.08},
    ),
]
DEFAULT_RATES = {
    "input": 3.00,
    "output": 15.00,
    "cache_write": 3.75,
    "cache_read": 0.30,
}


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
    # Split cache writes into 5m and 1h tiers when the breakdown is present.
    # 1h cache writes are billed at 2x base input, 5m at 1.25x (rates["cache_write"]).
    cc = usage.get("cache_creation") or {}
    cw_5m = cc.get("ephemeral_5m_input_tokens", 0)
    cw_1h = cc.get("ephemeral_1h_input_tokens", 0)
    if cw_5m == 0 and cw_1h == 0:
        cw_5m = usage.get("cache_creation_input_tokens", 0)
    cw_1h_rate = rates["input"] * 2.0
    return (
        usage.get("input_tokens", 0) / m * rates["input"]
        + usage.get("output_tokens", 0) / m * rates["output"]
        + cw_5m / m * rates["cache_write"]
        + cw_1h / m * cw_1h_rate
        + usage.get("cache_read_input_tokens", 0) / m * rates["cache_read"]
    )
