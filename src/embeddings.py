import re
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()
logger.add("embeddings.log", format="{time} {message}")


# parameters
MODEL_NAME = "MiMe-MeMo/MeMo-BERT-03"
MAX_TOKENS = 512
model = SentenceTransformer(MODEL_NAME)


def clean_whitespace(text):
    # rm newline characters
    text = text.replace('\n', ' ')
    # multiple spaces -> single space
    text = re.sub(r'\s+', ' ', text)
    # rm spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # rm excess spaces after punctuation (.,!? etc.)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    # leading and trailing spaces
    text = text.strip()
    
    return text


def simple_sentencize(text):
    """Split sentences on punctuation (and keep it)
    """
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def chunk_sentences(sentences, max_tokens=MAX_TOKENS):
    """Chunk sentences into as large as possible groups, while total length is bellow `max_tokens`.
    """
    output = []
    current_chunk = []
    chunk_len = 0

    for sentence in sentences:
        tokens = model.tokenize(sentence)
        seq_len = len(tokens["input_ids"])

        if chunk_len + seq_len > max_tokens:
            # edge case: sequence exceeds max_tokens & current chunk is empty
            if len(current_chunk) == 0: 
                parts = split_long_sentence(sentence, max_tokens)
                output.extend(parts)
            # else, append sentences from last iterations and start a new chunk 
            # for current text
            else:
                output.append(" ".join(current_chunk))
                current_chunk = []
                chunk_len = 0

        current_chunk.append(sentence)
        chunk_len += seq_len
    
    # append the last chunk
    if current_chunk:
        output.append(" ".join(current_chunk))

    return output


def split_long_sentence(sentence, max_tokens=MAX_TOKENS):
    """Tokenize word for word in case a single sentence is above context length.
    """
    words = sentence.split()
    parts = []
    current_part = []
    current_len = 0

    for word in words:
        tokens = model.tokenize(word)
        seq_len = len(tokens["input_ids"])
        
        if current_len + seq_len > max_tokens:
            parts.append(" ".join(current_part))
            current_part = []
            current_length = 0
        
        current_part.append(word)
        current_length += seq_len
    
    if current_part:
        parts.append(" ".join(current_part))

    return parts


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "Danish 19c novels KU-corpus",
    output_path: Path = INTERIM_DATA_DIR / "MeMo_embedding_chunks",
):

    # input & inference
    files = [item for item in input_path.iterdir()]
    processed_novels = []

    for path in tqdm(files):

        with open(path, "r") as fin:
            novel = fin.read()
        novel_name = path.stem

        # preprocessing
        try:
            novel_cl = clean_whitespace(novel)
            novel_sents = simple_sentencize(novel_cl)
            chunks = chunk_sentences(novel_sents)
        except Exception as e:
            logger.error(f"Prep {path}: {e}")

        # inference
        try:
            embs = []
            for chunk in chunks:
                emb = model.encode(chunk)
                embs.append(emb)
        except Exception as e:
            logger.error(f"Inference: {path}: {e}")
        
        # stashing
        processed_novels.append({
            "novel": novel_name,
            "chunk": chunks,
            "embedding": embs})

    # export
    dataset = Dataset.from_list(processed_novels)
    dataset.save_to_disk(output_path)


if __name__ == "__main__":
    app()
