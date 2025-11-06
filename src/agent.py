import json, numpy as np
from src.cluster_profiler import cluster_profiler
from src.rule_checker import rule_checker
from src.llm_utils import call_llama

def agent_for_cluster(cluster_id, df_clustered, index, embeddings, meta, numeric_cols, llm, top_k=3):
    profile = cluster_profiler(cluster_id, df_clustered, numeric_cols)
    query = f"Cluster with median LTV {profile['median_ltv']:.0f}, churn {profile['churn_rate']:.2f}, engagement {profile['avg_engagement']:.2f}"
    query_vec = np.mean(embeddings, axis=0).astype('float32').reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    retrieved = [{"doc_id": meta[i]["doc_id"], "text": meta[i]["text"], "score": float(D[0][j])} for j, i in enumerate(I[0])]
    prompt = f"""
You are a privacy-safe marketing strategist. 
Use the retrieved docs and the cluster profile below to generate retention/upsell insight.
Do NOT include or mention any numeric values from the cluster profile (even approximations).
Use qualitative terms like 'high', 'moderate', 'low' instead of numbers.

Cluster Profile: {json.dumps(profile, indent=2)}
Top Docs: {[r['text'] for r in retrieved]}

Return a JSON with:
{{
"summary": "...",
"recommended_action": "...",
"email_subject": "...",
"email_body_snippet": "... (must contain disclaimer 'This message is for informational purposes only.')",
"docs_used": [list of doc_ids]
}}
"""
    llm_output = call_llama(llm, prompt)
    try:
        parsed = json.loads(llm_output)
    except:
        parsed = {"summary": llm_output[:500], "recommended_action": None}
    rules = rule_checker(json.dumps(parsed), profile)
    result = {
        "cluster_id": cluster_id,
        "profile": profile,
        "retrieved_docs": retrieved,
        "llm_raw": llm_output,
        "parsed": parsed,
        "docs_used": parsed.get("docs_used", []),
        "rule_violations": rules,
        "quality_flag": "ok" if not rules else "review_required"
    }
    return result
