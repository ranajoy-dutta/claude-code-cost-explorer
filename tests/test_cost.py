from cost import calculate_cost, get_rates, DEFAULT_RATES


class TestGetRates:
    def test_opus_model(self):
        rates = get_rates("claude-opus-4-6")
        assert rates["input"] == 15.00 and rates["output"] == 75.00

    def test_sonnet_with_date_suffix(self):
        rates = get_rates("claude-sonnet-4-5-20250929")
        assert rates["input"] == 3.00

    def test_haiku_with_date_suffix(self):
        rates = get_rates("claude-haiku-4-5-20251001")
        assert rates["input"] == 0.80 and rates["output"] == 4.00

    def test_unknown_model_returns_default(self):
        assert get_rates("claude-unknown") == DEFAULT_RATES

    def test_empty_string_returns_default(self):
        assert get_rates("") == DEFAULT_RATES

    def test_none_returns_default(self):
        assert get_rates(None) == DEFAULT_RATES

    def test_synthetic_model_returns_default(self):
        assert get_rates("<synthetic>") == DEFAULT_RATES


class TestCalculateCost:
    def test_zero_usage(self):
        assert calculate_cost("claude-sonnet-4-6", {}) == 0.0

    def test_none_usage(self):
        assert calculate_cost("claude-sonnet-4-6", None) == 0.0

    def test_output_tokens_sonnet(self):
        # 1M output tokens at $15/MTok = $15
        assert (
            abs(
                calculate_cost("claude-sonnet-4-6", {"output_tokens": 1_000_000}) - 15.0
            )
            < 1e-9
        )

    def test_input_tokens_opus(self):
        # 1M input tokens at $15/MTok = $15
        assert (
            abs(calculate_cost("claude-opus-4-6", {"input_tokens": 1_000_000}) - 15.0)
            < 1e-9
        )

    def test_cache_write_haiku(self):
        assert (
            abs(
                calculate_cost(
                    "claude-haiku-4-5-20251001",
                    {"cache_creation_input_tokens": 1_000_000},
                )
                - 1.00
            )
            < 1e-9
        )

    def test_cache_read_haiku(self):
        assert (
            abs(
                calculate_cost(
                    "claude-haiku-4-5-20251001", {"cache_read_input_tokens": 1_000_000}
                )
                - 0.08
            )
            < 1e-9
        )

    def test_all_token_types_combined(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 200,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 400,
        }
        expected = (100 * 3 + 200 * 15 + 300 * 3.75 + 400 * 0.30) / 1e6
        assert abs(calculate_cost("claude-sonnet-4-6", usage) - expected) < 1e-9

    def test_cost_is_positive(self):
        assert (
            calculate_cost(
                "claude-opus-4-6", {"input_tokens": 500, "output_tokens": 100}
            )
            > 0
        )
