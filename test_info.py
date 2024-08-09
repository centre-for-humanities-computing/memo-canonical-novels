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
from tslearn.metrics import dtw
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
# louise's dtw script
def calc_signal_difference(no_artist_signal_df, org_signal, w_size):

    # convert the novelty signal excluding a specific artist to a list
    n_df = no_artist_signal_df.iloc[w_size:, :]
    n_hat = n_df['N_hat'].tolist()

    # do the same for the original novelty signal
    n_df_original = org_signal.iloc[w_size:, :]
    n_hat_original = n_df_original['N_hat'].tolist()

    # calculate dtw (Dynamic Time Warping) difference
    dtw_score_n = dtw(n_hat, n_hat_original)

    # convert the resonance signal excluding a specific artist to a list
    r_df = no_artist_signal_df.iloc[w_size:-w_size, :]
    r_hat = r_df['R_hat'].tolist()

    # do the same for the original resonance signal
    r_df_original = org_signal.iloc[w_size:-w_size, :]
    r_hat_original = r_df_original['R_hat'].tolist()

    # calculate difference
    dtw_score_r = dtw(r_hat, r_hat_original)

    return dtw_score_n, dtw_score_r


def calc_dtw_scores(ds, artist, w_size, org_signal):

    # remove artist from data
    ds_no_artist = remove_artist(ds, artist)

    # calculate new signal with artist removed
    signal_df = calc_signals(ds_no_artist, w_size)

    # calculate dtw scores for novelty and resonance for the new and original signal
    dtw_score_n, dtw_score_r = calc_signal_difference(signal_df, org_signal, w_size)

    return dtw_score_n, dtw_score_r


# %%
dtw_n = []
dtw_r = []

for artist in grouped_df['artist']:
    dtw_scores = calc_dtw_scores(ds, artist, 20, org_signal) # using window size of 20

    dtw_n.append(dtw_scores[0])
    dtw_r.append(dtw_scores[1])