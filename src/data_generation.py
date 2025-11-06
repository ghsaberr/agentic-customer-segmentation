import numpy as np
import pandas as pd

def generate_insurance_dataset_realistic_v4(
    n_customers: int = 5000,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    correlation_strength: float = 0.6,
    return_metadata: bool = False,
    adaptive_noise: bool = True,
    base_std: float = 0.05,
    strength: float = 0.08
) -> pd.DataFrame:
    """
    Version 4 — Realistic Synthetic Insurance Dataset Generator with Adaptive Noise
    
    ✅ Enhancements over v3:
    - Integrated adaptive behavioral noise based on income and age
    - Slightly more realistic renewal/churn variation by customer profile
    - Optional automatic saving and metadata return
    """

    rng = np.random.default_rng(seed)
    correlation_strength = np.clip(correlation_strength, 0.0, 0.95)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # --- 1. Demographics (age–income correlation)
    mean = [0, 0]
    cov = [[1, correlation_strength], [correlation_strength, 1]]
    z_age, z_income = rng.multivariate_normal(mean, cov, n_customers).T

    ages = np.clip((z_age * 12 + 45).astype(int), 18, 85)
    income_levels = np.exp(z_income * 0.3) * (25000 + 600 * ages)
    income_levels *= rng.lognormal(0, 0.2, n_customers)
    income_levels = np.clip(income_levels, 20000, 250000)

    genders = rng.choice(["M", "F"], n_customers, p=[0.52, 0.48])
    regions = rng.choice(["Urban", "Suburban", "Rural"], n_customers, p=[0.4, 0.45, 0.15])

    # --- 2. Policy Type (income-driven probability)
    income_scaled = (income_levels - income_levels.min()) / (income_levels.max() - income_levels.min())
    policy_probs = np.vstack([
        np.clip([
            0.55 - 0.10 * income_scaled,
            0.25 + 0.10 * income_scaled,
            0.10 + 0.08 * income_scaled,
            np.full_like(income_scaled, 0.10)
        ], 0, None).T for _ in range(1)
    ]).reshape(n_customers, 4)
    policy_probs /= policy_probs.sum(axis=1, keepdims=True)
    policy_types = np.array(["Auto", "Home", "Life", "Health"])
    chosen_policy = [rng.choice(policy_types, p=p) for p in policy_probs]

    # --- 3. Premium Amounts (policy + income)
    base_premiums = {"Auto": 800, "Home": 1600, "Life": 2000, "Health": 1200}
    premium_base = np.array([base_premiums[p] for p in chosen_policy])
    premium_amount = premium_base * (1 + np.sqrt(income_levels / 150000) * 0.5)
    premium_amount *= rng.lognormal(0, 0.25, n_customers)
    premium_amount = np.clip(premium_amount, 300, 10000)

    # --- 4. Policy tenure
    tenure_months = np.clip(rng.exponential(30, n_customers), 3, 84)
    tenure_days = (tenure_months * 30).astype(int)
    valid_start_days = np.maximum(1, (end - start).days - tenure_days)
    start_offsets = rng.integers(0, valid_start_days, n_customers)
    policy_start_dates = start + pd.to_timedelta(start_offsets, unit="D")
    policy_end_dates = policy_start_dates + pd.to_timedelta(tenure_days, unit="D")
    policy_end_dates = policy_end_dates.where(policy_end_dates <= pd.Timestamp(end), pd.Timestamp(end))

    # --- 5. Payment behavior
    payment_freq = rng.choice(["Monthly", "Quarterly", "Annual"], n_customers, p=[0.6, 0.25, 0.15])
    payment_regularity = np.clip(rng.beta(4, 1.5, n_customers) + rng.normal(0, 0.05, n_customers), 0.3, 1.0)

    # --- 6. Claims (depends on age + policy)
    base_claim_rate = np.where(ages < 25, 0.25, np.where(ages > 65, 0.18, 0.1))
    policy_factor = np.array([{"Auto":1.4, "Home":0.8, "Life":0.6, "Health":1.0}[p] for p in chosen_policy])
    claim_lambda = base_claim_rate * policy_factor * np.sqrt(tenure_months / 12)
    num_claims = np.clip(rng.poisson(claim_lambda), 0, 15)

    # Claim costs — lognormal by policy type
    claim_cost_params = {"Auto": (7.8, 0.9), "Home": (8.3, 0.8), "Life": (9.0, 0.6), "Health": (8.0, 1.0)}
    claim_costs = np.zeros(n_customers)
    for p, (mean, sigma) in claim_cost_params.items():
        idx = np.array(chosen_policy) == p
        claim_costs[idx] = num_claims[idx] * rng.lognormal(mean, sigma, idx.sum())

    # --- 7. Renewal probability (with adaptive noise)
    logits = (
        -0.02 * (ages - 45)
        - 0.3 * num_claims
        + 1.0 * payment_regularity
        + 0.004 * tenure_months
        + rng.normal(0, 0.5, n_customers)
    )
    renewal_prob = 1 / (1 + np.exp(-logits))
    renewal_prob = np.clip(renewal_prob, 0.05, 0.95)

    # 🔹 Adaptive noise based on age & income
    if adaptive_noise:
        income_norm = (income_levels - income_levels.min()) / (income_levels.max() - income_levels.min())
        age_norm = (ages - ages.min()) / (ages.max() - ages.min())

        # Stable customers: older + higher income
        stability_score = (income_norm + age_norm) / 2
        noise_std = base_std + strength * (1 - stability_score)

        adaptive_noise = rng.normal(0, noise_std)
        renewal_prob = np.clip(renewal_prob + adaptive_noise, 0.01, 0.99)

    churned = rng.binomial(1, 1 - renewal_prob)

    # --- 8. Last claim date
    has_claims = num_claims > 0
    last_claim_date = np.full(n_customers, np.datetime64("NaT"), dtype="datetime64[ns]")
    random_offsets = rng.exponential(180, has_claims.sum()).astype(int)
    temp_dates = policy_end_dates[has_claims] - pd.to_timedelta(random_offsets, unit="D")
    temp_dates = np.maximum(temp_dates.values.astype("datetime64[D]"), policy_start_dates[has_claims].values.astype("datetime64[D]"))
    last_claim_date[has_claims] = temp_dates

    # --- 9. Lifetime Value (LTV)
    expected_value = premium_amount * (tenure_months / 12)
    claim_penalty = np.minimum(claim_costs / (expected_value + 1e-6), 0.8)
    ltv = expected_value * (1 - claim_penalty) * (0.8 + 0.4 * renewal_prob)
    ltv[churned == 1] *= rng.uniform(0.4, 0.8, churned.sum())
    ltv = np.clip(ltv, 0, 250_000)

    # --- 10. Derived temporal features
    policy_age_days = (policy_end_dates - policy_start_dates).days
    days_since_last_claim = (policy_end_dates - last_claim_date).days.astype(float).to_numpy()
    days_since_last_claim[~has_claims] = np.nan
    is_active = (policy_end_dates >= end).astype(int)

    # --- 11. Final DataFrame
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(1, n_customers + 1)],
        "age": ages,
        "gender": genders,
        "region": regions,
        "income_level": np.round(income_levels, 2),
        "policy_type": chosen_policy,
        "premium_amount": np.round(premium_amount, 2),
        "payment_frequency": payment_freq,
        "payment_regularity": np.round(payment_regularity, 3),
        "tenure_months": np.round(tenure_months, 1),
        "num_claims": num_claims,
        "claim_costs": np.round(claim_costs, 2),
        "policy_start_date": policy_start_dates,
        "policy_end_date": policy_end_dates,
        "last_claim_date": pd.to_datetime(last_claim_date),
        "renewal_probability": np.round(renewal_prob, 3),
        "churned": churned,
        "customer_ltv": np.round(ltv, 2),
        "policy_age_days": policy_age_days,
        "days_since_last_claim": days_since_last_claim,
        "is_active": is_active,
    })

    if return_metadata:
        metadata = {
            "n_customers": n_customers,
            "start_date": start_date,
            "end_date": end_date,
            "seed": seed,
            "correlation_strength": correlation_strength,
            "adaptive_noise": adaptive_noise,
            "generated_on": pd.Timestamp.now().isoformat(),
        }
        return df, metadata

    return df

import pandas as pd
import numpy as np

def augment_with_engagement_and_flags_v3(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    ✅ Version 3 — Realistic augmentation of insurance dataset with engagement, delinquency, and risk flags.
    
    Key improvements:
    - No leakage from churn/renewal into behavior features.
    - Behavioral realism: digital users (younger, urban, higher income) show more engagement.
    - Payment and claim behaviors influence risk/delinquency flags probabilistically.
    - Adds derived scores and flags used for segmentation.
    """

    rng = np.random.default_rng(seed)
    df = df.copy()

    # --- 1. Engagement behavior (portal logins, calls, emails)
    # Digital affinity by region & age
    digital_affinity = (
        0.6 * (df["region"] == "Urban").astype(float)
        + 0.3 * (df["region"] == "Suburban").astype(float)
        + 0.1 * (df["region"] == "Rural").astype(float)
    )
    digital_affinity *= np.clip((70 - df["age"]) / 40, 0, 1)  # Younger = more digital
    digital_affinity *= np.clip(np.log1p(df["income_level"]) / np.log(150_000), 0.6, 1.2)

    # Engagement base (mean ~ digital_affinity × tenure)
    base_logins = 3 + 15 * digital_affinity + 0.05 * df["tenure_months"]
    portal_logins = np.clip(rng.normal(base_logins, 2), 0, None).astype(int)

    # Support calls (older + rural → more)
    support_calls = np.clip(
        rng.poisson(1 + 0.015 * (df["age"] - 30) + 1.5 * (df["region"] == "Rural").astype(int)),
        0,
        15,
    )

    # Emails opened (proportional to digital use)
    emails_opened = np.clip(rng.poisson(10 * digital_affinity + 2), 0, 50)

    # --- 2. Payment delinquency
    base_delinquency_prob = (
        0.04 + 0.08 * (df["payment_regularity"] < 0.8).astype(float)
        + 0.03 * (df["income_level"] < 40_000).astype(float)
        + 0.02 * (df["policy_type"] == "Auto").astype(float)
    )
    delinquency_count = rng.binomial(n=3, p=np.clip(base_delinquency_prob, 0, 0.8))

    # --- 3. Engagement score (normalized composite)
    max_portal, max_emails, max_calls = (
        portal_logins.max(),
        emails_opened.max(),
        support_calls.max(),
    )
    engagement_score = (
        0.5 * (portal_logins / (max_portal + 1e-5))
        + 0.3 * (emails_opened / (max_emails + 1e-5))
        + 0.2 * (1 - support_calls / (max_calls + 1e-5))
    )
    engagement_score = np.clip(engagement_score, 0, 1)

    # --- 4. Risk flags (no leakage)
    # High claim frequency/severity → high risk
    claim_freq_flag = (df["num_claims"] > 3).astype(int)
    high_claim_cost_flag = (df["claim_costs"] > df["claim_costs"].median() * 2).astype(int)

    # Financial risk from delinquency
    payment_risk_flag = (delinquency_count >= 2).astype(int)

    # Behavior risk (low engagement)
    low_engagement_flag = (engagement_score < 0.3).astype(int)

    # --- 5. Lifecycle and product mix
    # Simple heuristic: stage inferred from tenure
    lifecycle_stage = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 36, 72, np.inf],
        labels=["New", "Growth", "Mature", "Legacy"],
        right=False,
    )

    # Product mix: probability of having another product (income + engagement)
    multi_policy_prob = np.clip(0.1 + 0.001 * (df["income_level"] / 1000) + 0.3 * engagement_score, 0, 0.9)
    multi_policy_flag = rng.binomial(1, multi_policy_prob)

    # --- 6. Engagement trends (rolling proxy)
    recent_logins_3m = np.clip(
        (portal_logins * rng.uniform(0.6, 1.0, len(df))).astype(int),
        0,
        portal_logins,
    )
    engagement_trend = np.where(
        recent_logins_3m > (0.8 * portal_logins), "Up",
        np.where(recent_logins_3m < (0.5 * portal_logins), "Down", "Stable")
    )

    # --- 7. Add all columns
    df["portal_logins"] = portal_logins
    df["support_calls"] = support_calls
    df["emails_opened"] = emails_opened
    df["delinquency_count"] = delinquency_count
    df["engagement_score"] = np.round(engagement_score, 3)
    df["claim_freq_flag"] = claim_freq_flag
    df["high_claim_cost_flag"] = high_claim_cost_flag
    df["payment_risk_flag"] = payment_risk_flag
    df["low_engagement_flag"] = low_engagement_flag
    df["multi_policy_flag"] = multi_policy_flag
    df["lifecycle_stage"] = lifecycle_stage
    df["engagement_trend"] = engagement_trend

    return df