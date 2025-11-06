def rule_checker(output_text, profile):
    rules = []
    for term in ["income", "ltv", "salary", "spend"]:
        if str(int(profile.get("median_ltv",0))) in output_text:
            rules.append("leak: median_ltv")
        if f"{profile.get('avg_engagement',0):.2f}" in output_text:
            rules.append("leak: avg_engagement")
    if any(keyword in output_text.lower() for keyword in ["dear [customer name]", "based on your income"]):
        rules.append("pii_leak_detected")
    if "informational purposes only" not in output_text.lower():
        rules.append("missing_disclaimer")
    return rules
