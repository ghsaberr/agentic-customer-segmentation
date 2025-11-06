from src.rule_checker import rule_checker

def test_rule_checker_detects_missing_disclaimer():
    profile = {"median_ltv": 5000, "avg_engagement": 0.5}
    text_missing = "This is a suggested message without disclaimer"
    res = rule_checker(text_missing, profile)
    assert any("missing_disclaimer" in r for r in res), f"Expected missing_disclaimer, got {res}"

    text_with_disclaimer = "Hello. (This message is for informational purposes only.)"
    res2 = rule_checker(text_with_disclaimer, profile)
    assert all("missing_disclaimer" not in r for r in res2)
