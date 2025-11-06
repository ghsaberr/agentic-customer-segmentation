import pandas as pd
import numpy as np
import random
import uuid
from datetime import timedelta
from textwrap import shorten

def generate_document_corpus_v2(
    customers_df: pd.DataFrame = None,
    n_docs: int = 60,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    verbose: bool = False,
    max_passage_words: int = 200,
) -> pd.DataFrame:
    """
    Generate a synthetic document corpus aligned with a customer/policy dataset.

    Features:
      - Optional linking to customers_df: attaches policy_id / customer_id where relevant
      - Category/subtype normalized across all docs
      - created_date sampled between start_date and end_date (can align to policy dates)
      - Topic-aware keywords and longer contextual text for better retrieval
      - Passage splitting (if text long) and stable doc_id / passage_id scheme
      - Returns DataFrame ready for embedding + indexing with metadata fields:
        ['doc_id', 'passage_id', 'policy_id', 'customer_id', 'category', 'subtype',
         'source', 'created_date', 'text', 'keywords']
    """
    np.random.seed(seed)
    random.seed(seed)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    total_days = max(1, (end - start).days)

    # --- Helper maps for realistic keyword selection
    kw_map = {
        "Auto": ["auto", "vehicle", "driver", "collision", "telematics", "premium"],
        "Home": ["home", "property", "dwelling", "flood", "fire", "repair", "inspection"],
        "Life": ["life", "beneficiary", "term", "payout", "underwriting", "medical"],
        "Health": ["health", "hospital", "claim", "copay", "provider", "network"],
        "General": ["renewal", "notice", "payment", "correspondence", "policy", "update"],
        "Underwriting": ["risk", "score", "medical", "credit", "assessment"],
        "CaseSummary": ["case", "resolution", "retention", "offer", "outcome"],
        "Product": ["product", "plan", "feature", "tier", "update", "launch"],
        "Marketing": ["campaign", "offer", "discount", "promotion", "customer", "communication"],
    }

    # --- Base templates (policy, underwriting, product, marketing, etc.)
    base_templates = [
        ("Auto", "Policy Wording",
         "This Auto policy includes comprehensive, collision and liability coverages. Deductibles and premium adjustments depend on driver age, claim history, and telematics scores. {extra}"),
        ("Home", "Policy Wording",
         "This Home policy covers dwelling, personal property and loss-of-use. Coverage limits and exclusions depend on construction type and local hazards. {extra}"),
        ("Life", "Policy Wording",
         "Term and whole-life options are available. Premiums and eligibility depend on medical underwriting and beneficiary designation. {extra}"),
        ("Health", "Policy Wording",
         "Tiered health plans (Bronze/Silver/Gold) with varying provider networks and out-of-pocket limits. Pre-authorization required for selected services. {extra}"),

        ("General", "Renewal Notice",
         "Your policy is due for renewal on {policy_end_date}. Please review updated premium and coverage information. {extra}"),
        ("General", "Premium Notice",
         "We are notifying you of a premium change effective {effective_date} due to regional claim trends. {extra}"),
        ("Underwriting", "Underwriting Note",
         "Underwriting analysis: risk factors include {risk_factors}. Recommends class change: {recommendation}. {extra}"),
        ("CaseSummary", "Retention Case",
         "Case summary: Customer was at risk due to recent claims; retention offer included a targeted discount and service outreach. Outcome: {outcome}. {extra}"),
        ("Product", "Product Update",
         "Product update: New telematics discount launched. Eligible customers will receive communication and opt-in instructions. {extra}"),
        ("General", "Claim Update",
         "Claim {claim_ref} is being processed. Current status: {status}. Estimated settlement: {estimate}. {extra}"),
    ]

    # --- Add new content types: Product Brochures & Campaign Templates
    base_templates.extend([
        ("Product", "Brochure",
         "Introducing our {policy_type} insurance plan — designed to provide better protection and flexible coverage options. Key highlights include: {highlights}. Learn how you can save with telematics and bundled discounts. {extra}"),
        ("Product", "Product Overview",
         "Our latest {policy_type} policy offers enhanced benefits, simplified claims, and faster online servicing. Whether you’re a new or existing customer, this plan helps you manage your coverage efficiently. {extra}"),
        ("Marketing", "Campaign Template",
         "Subject: {subject}\n\nDear customer, we’re excited to announce our new {policy_type} plan. {benefit_statement}. Don’t miss this opportunity — apply by {deadline}. {cta}"),
        ("Marketing", "Email Campaign",
         "Get rewarded for staying protected! Our loyalty campaign offers up to 15% off renewals on select {policy_type} policies. {extra} Join thousands who already switched and saved."),
    ])

    # --- Prepare synthetic linking to customers/policies if available
    if customers_df is not None and "policy_type" in customers_df.columns:
        customers_df = customers_df.copy()
        if "policy_id" not in customers_df.columns:
            customers_df["policy_id"] = [f"P{100000 + i}" for i in range(len(customers_df))]
        policy_links = customers_df[["customer_id", "policy_id", "policy_type", "policy_start_date", "policy_end_date"]].to_dict("records")
    else:
        policy_links = [{"customer_id": None, "policy_id": None, "policy_type": random.choice(["Auto","Home","Life","Health"]),
                         "policy_start_date": start + timedelta(days=random.randint(0, total_days//2)),
                         "policy_end_date": start + timedelta(days=random.randint(total_days//2, total_days))} for _ in range(200)]

    # --- Instantiate docs
    all_docs = []
    for (category, subtype, template) in base_templates:
        for _ in range(3):  # multiple variants per template
            extra_phrases = [
                "This language was updated to reflect regulatory guidance.",
                "This clause applies only to new business issued within the last 12 months.",
                "Refer to underwriting manual section 4.2 for scoring rules.",
                "This summary reflects the current version of the product.",
            ]
            extra = random.choice(extra_phrases)

            safe_kwargs = dict(
                extra=extra,
                policy_type=random.choice(["Auto", "Home", "Life", "Health"]),
                policy_end_date="2024-12-31",
                effective_date="2024-01-01",
                risk_factors="driver age, credit score",
                recommendation="standard",
                outcome="retained",
                claim_ref=f"CLM{random.randint(10000,99999)}",
                status="in progress",
                estimate="$1500",
                highlights=random.choice([
                    "customizable deductibles, multi-policy savings, and digital servicing tools",
                    "expanded coverage for extreme weather events and enhanced roadside assistance",
                    "wellness rewards and lower premiums for safe drivers",
                ]),
                subject=random.choice([
                    "Save More with Our New Plan!",
                    "Exclusive Renewal Offer Inside",
                    "Your Coverage, Upgraded",
                ]),
                benefit_statement=random.choice([
                    "Enjoy lower premiums and smarter coverage options tailored to your needs",
                    "Upgrade to enhanced protection with minimal paperwork",
                    "Protect what matters most with our simplified digital policies",
                ]),
                deadline=random.choice(["June 30", "September 1", "December 31"]),
                cta=random.choice(["Click here to learn more.", "Enroll today!", "Get your personalized quote now."]),
            )

            body = template.format(**safe_kwargs)
            all_docs.append({"category": category, "subtype": subtype, "text": body})

    # --- Expand if fewer than n_docs
    while len(all_docs) < n_docs:
        sample = random.choice(all_docs)
        variation = sample.copy()
        rev = random.randint(1, 5)
        change = random.choice(["Updated rates", "Coverage change", "New clause added", "Review summary", "Clarified exclusions"])
        variation["text"] = f"{variation['text']} (Revision {rev}: {change})."
        all_docs.append(variation)

    # --- Balanced sampling
    if len(all_docs) > n_docs:
        categories = list(set([d["category"] for d in all_docs]))
        chosen = []
        for c in categories:
            candidates = [d for d in all_docs if d["category"] == c]
            chosen.append(random.choice(candidates))
        remaining = n_docs - len(chosen)
        others = [d for d in all_docs if d not in chosen]
        chosen.extend(random.sample(others, remaining))
        all_docs = chosen

    # --- Build final corpus with metadata and passage splitting
    corpus = []
    for i, doc in enumerate(all_docs, start=1):
        uid = uuid.uuid4().hex[:8].upper()
        doc_id = f"DOC_{uid}"
        link = random.choice(policy_links)
        linked_policy = link["policy_id"] if random.random() < 0.4 else None
        linked_customer = link["customer_id"] if linked_policy is not None else None

        if linked_policy and link.get("policy_start_date") is not None and link.get("policy_end_date") is not None:
            p_start = pd.to_datetime(link["policy_start_date"])
            p_end = pd.to_datetime(link["policy_end_date"])
            if pd.isna(p_start) or pd.isna(p_end) or p_end <= p_start:
                created_date = start + timedelta(days=random.randint(0, total_days))
            else:
                created_date = p_start + timedelta(days=random.randint(0, max(1, (p_end - p_start).days)))
        else:
            created_date = start + timedelta(days=random.randint(0, total_days))

        source = "internal" if doc["subtype"] in ("Underwriting Note", "Agent Note", "CaseSummary") else random.choice(["customer-facing", "internal", "agent-note"])
        kw_pool = kw_map.get(doc["category"], kw_map["General"])
        keywords = random.sample(kw_pool, k=min(len(kw_pool), random.randint(3, 5)))

        context_sentences = [
            "This excerpt summarizes the key points relevant to pricing and retention.",
            "Please refer to the full policy for exact conditions and exclusions.",
            "Contact the underwriting or marketing team for clarifications.",
            "This note was generated as part of an automated review process."
        ]
        text = doc["text"] + " " + " ".join(random.sample(context_sentences, k=random.randint(1, 2)))
        text = shorten(text, width=800, placeholder=" ...")

        words = text.split()
        passages = [text] if len(words) <= max_passage_words else [
            " ".join(words[i:i+max_passage_words]) for i in range(0, len(words), max_passage_words)
        ]

        for p_idx, passage in enumerate(passages, start=1):
            passage_id = f"{doc_id}_P{p_idx:02d}"
            corpus.append({
                "doc_id": doc_id,
                "passage_id": passage_id,
                "policy_id": linked_policy,
                "customer_id": linked_customer,
                "category": doc["category"],
                "subtype": doc["subtype"],
                "source": source,
                "created_date": created_date,
                "created_date_str": created_date.strftime("%Y-%m-%d"),
                "text": passage,
                "keywords": keywords,
            })

    corpus_df = pd.DataFrame(corpus)

    if verbose:
        print(f"✅ Generated document corpus: {len(corpus_df)} passages from {len(all_docs)} docs")
        print(f"Categories: {sorted(corpus_df['category'].unique().tolist())}")
        print("Example passage metadata:")
        print(corpus_df.iloc[0].to_dict())

    return corpus_df
