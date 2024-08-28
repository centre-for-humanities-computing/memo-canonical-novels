from sentence_transformers import SentenceTransformer

def count_tokens_in_prompt(model_name: str, prompt: str) -> int:
    """Count how many tokens will the prompt be tokenized into.
    Disclaimer: depending on model architecture, there will be special tokens added anyway,
    so your final prompt sequence length might be lower.
    """
    model = SentenceTransformer(model_name)
    tokens = model.tokenize(prompt)
    seq_len = len(tokens["input_ids"])
    return seq_len
