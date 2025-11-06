def call_llama(llm, prompt):
    """Privacy-safe LLaMA chat wrapper."""
    try:
        messages = [
            {"role": "system", "content": "You are a privacy-safe marketing strategist."},
            {"role": "user", "content": prompt.strip()}
        ]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        output = response["choices"][0]["message"]["content"].strip()
        return output if output else "⚠️ LLM returned empty chat message."
    except Exception as e:
        return f"LLM error: {e}"
