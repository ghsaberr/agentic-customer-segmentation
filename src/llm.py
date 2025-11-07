from llama_cpp import Llama

def load_llm(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048, n_threads=6):
    """
    Loads quantized mistral model locally with llama_cpp
    """
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        temperature=0.2,
        seed=42,
        verbose=False
    )
    return llm
