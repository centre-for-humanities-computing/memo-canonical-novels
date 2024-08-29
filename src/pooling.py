from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from datasets import Dataset
import numpy as np

from src.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def mean_pooling(dataset: Dataset):
    """
    """
    out = []
    for novel in tqdm(dataset):
        chunk_embs = novel["embedding"]
        emb = np.mean(chunk_embs, axis=0)
        out.append(emb)

    return out


@app.command()
def main(
    input_ds: str = typer.Argument(help="name of dataset in INTERIM_DATA_DIR to process")
):

    input_path: Path = INTERIM_DATA_DIR / input_ds
    output_ds = input_ds.replace("emb__", "meanpool__")
    output_path: Path = PROCESSED_DATA_DIR / output_ds

    ds_chunks = Dataset.load_from_disk(input_path)
    n_chunks = [len(chunks) for chunks in ds_chunks["chunk"]]

    # pooling, could be different options
    pooled_embs = mean_pooling(ds_chunks)

    # export
    ds_pooled = Dataset.from_dict({
        "novel": ds_chunks["novel"],
        "embedding": pooled_embs,
        "n_chunks_orig": n_chunks,
    })

    ds_pooled.save_to_disk(output_path)


if __name__ == "__main__":
    app()
