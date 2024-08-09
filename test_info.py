# %%
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.decomposition import PCA
from infodynamics import WindowedRollingDistance
from infodynamics.util import calc_vector_histogram

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# %%
input_path = PROCESSED_DATA_DIR / "MeMo_embeddings_pooled"
ds = Dataset.load_from_disk(input_path)

# %%
meta_path = RAW_DATA_DIR / "DANISH_CORPUS_METADATA.xlsx"
meta = pd.read_excel(meta_path)

# %%
dims = PCA(n_components=2).fit_transform(ds["embedding"])

# %%
plt.plot(dims[:, 0], dims[:, 1], ".")

# %%
# order the novels
years = [int(name[0:4]) for name in ds["novel"]]
ds_sorted = ds.add_column("year", years).sort("year")

# %% 
# infodynamics signal
X_e = np.array(ds_sorted["embedding"])
X_h = np.array([calc_vector_histogram(emb) for emb in ds_sorted["embedding"]])

ntr_jsd_20 = WindowedRollingDistance(
    measure="jensenshannon",
    window_size=20,
    ).fit(X_h)

ntr_cos_20 = WindowedRollingDistance(
    measure="cosine",
    window_size=20,
    ).fit(X_e)

ntr_cos_2 = WindowedRollingDistance(
    measure="cosine",
    window_size=2,
    ).fit(X_e)

# %%
plt.plot(ntr_jsd_20.signal["N_hat"])

# %%
plt.plot(ntr_cos_20.signal["N_hat"])

# %%
### compare to all novels
from sklearn.metrics import pairwise_distances

D = pairwise_distances(X_e)

# %%
import seaborn as sns

sns.heatmap(D, cmap="mako")

# %%
