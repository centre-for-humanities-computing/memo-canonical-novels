import pandas as pd
import json
from datasets import Dataset

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import spearmanr

import joypy

import plotly.express as px
import os

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity


from sklearn.decomposition import PCA

from venny4py.venny4py import *
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_venn import venn3, venn3_circles

import re
from sklearn.feature_extraction.text import TfidfVectorizer

# for stylistics
from pathlib import Path

from lexical_diversity import lex_div as ld

import neurokit2 as nk
from nltk.tokenize import sent_tokenize, word_tokenize

import nltk
import spacy
from tqdm import tqdm

import textdescriptives as td

import bz2
import gzip

from collections import Counter
from math import log

from sklearn.utils import resample


##

def get_first_word(title):
    words = str(title).split(' ')
    if len(words) > 0 and len(words[0]) <= 10:
        return words[0]
    return words[0][:10]

# function to make dendrogram from embeddings
def plot_dendrogram(df, col_to_color, col_to_label, l, h, palette='Set2'):
    unique_categories = df[col_to_color].unique()

    # colors
    cat_map = dict(zip(df[col_to_label],df[col_to_color]))
    color_dict = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

    # prepare data for plotting
    embeddings_matrix = np.stack(df['EMBEDDING'].values)
    cosine_dist_matrix = cosine_distances(embeddings_matrix)
    
    if cosine_dist_matrix.shape[0] != cosine_dist_matrix.shape[1]:
        raise ValueError("Distance matrix is not square.")

    Z = linkage(cosine_dist_matrix, method='ward')

    # dendrogram plot
    sns.set_style('whitegrid')
    plt.figure(figsize=(l, h))
    dend = dendrogram(Z, labels=df[col_to_label].values, orientation='top', leaf_font_size=10, color_threshold=0, above_threshold_color='black')

    # Labels
    # get x-tick labels
    ax = plt.gca()
    xticklabels = ax.get_xticklabels()

    # apply colors labels
    used_colors = {}
    for tick in xticklabels:
        label = tick.get_text()
        # just to make sure we have no other labels in there
        if label in cat_map:
            value = cat_map[label]
            color = color_dict[value]
            tick.set_color(color)
            used_colors[value] = color
        else:
            tick.set_color('black')
    
    # layout
    plt.xlabel("Cosine Distance")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in used_colors.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    #plt.title(f'Canon/Historical tags in {len(df)} novels')
    plt.show()


# PCA plot

def plot_pca(ax, data, title, colormapping):
    # Handle embeddings
    embeddings_array = np.array(data["EMBEDDING"].to_list(), dtype=np.float32)
    
    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["CATEGORY"] = data["CATEGORY"].values


    #markers_dict = dict(zip(colormapping.keys(), ['o' if x != 'O' else '.' for x in colormapping.keys()])) # you could make a different marker for the Other category
    # We're gonna set a different alpha for the 'O' category
    alpha_dict = dict(zip(colormapping.keys(), [0.65 if x != 'O' else 0.2 for x in colormapping.keys()]))

    # Plot each category
    for category in df_pca["CATEGORY"].unique():
        subset = df_pca[df_pca["CATEGORY"] == category]

        #marker = markers_dict.get(category) 
        alpha = alpha_dict.get(category)
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=colormapping.get(category),
            label=category,
            alpha=alpha,
            edgecolor='black',
            s=120,
            marker='o' #marker
        )
    
    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in colormapping.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    ax.axis("equal")


# making functions for the comparing types of embeddings
# we need to pass something else than df['EMBEDDINGS'] - so that is the only difference from the ones above

# function to make dendrogram from embeddings
def plot_dendrogram_different_embeddings(df, cosine_matrix, col_to_color, col_to_label, l, h, method_name=None, palette='Set2'):
    unique_categories = df[col_to_color].unique()

    # colors
    cat_map = dict(zip(df[col_to_label],df[col_to_color]))
    color_dict = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

    # prepare data for plotting
    #embeddings_matrix = matrix #np.stack(df[data_col].values)

    cosine_dist_matrix = cosine_matrix #cosine_distances(embeddings_matrix)
    
    if cosine_dist_matrix.shape[0] != cosine_dist_matrix.shape[1]:
        raise ValueError("Distance matrix is not square.")

    Z = linkage(cosine_dist_matrix, method='ward')

    # dendrogram plot
    plt.figure(figsize=(l, h))
    dend = dendrogram(Z, labels=df[col_to_label].values, orientation='top', leaf_font_size=10, color_threshold=0, above_threshold_color='black')

    # Labels
    # get x-tick labels
    ax = plt.gca()
    xticklabels = ax.get_xticklabels()

    # apply colors labels
    used_colors = {}
    for tick in xticklabels:
        label = tick.get_text()
        # just to make sure we have no other labels in there
        if label in cat_map:
            value = cat_map[label]
            color = color_dict[value]
            tick.set_color(color)
            used_colors[value] = color
        else:
            tick.set_color('black')
    
    # layout
    plt.xlabel("Cosine Distance")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in used_colors.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    #plt.title(f'Canon/Historical tags in {len(df)} novels, label: {method_name}')
    plt.show()




def plot_pca_different_embeddings(ax, data, data_col, title, colormapping):
    # Handle embeddings
    embeddings_array = np.array(data[data_col].to_list(), dtype=np.float32)
    
    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["CATEGORY"] = data["CATEGORY"].values

    #markers_dict = dict(zip(colormapping.keys(), ['o' if x != 'O' else '.' for x in colormapping.keys()])) # you could make a different marker for the Other category
    # We're gonna set a different alpha for the 'O' category
    alpha_dict = dict(zip(colormapping.keys(), [0.65 if x != 'O' else 0.2 for x in colormapping.keys()]))

    # Plot each category
    for category in df_pca["CATEGORY"].unique():
        subset = df_pca[df_pca["CATEGORY"] == category]

        #marker = markers_dict.get(category) 
        alpha = alpha_dict.get(category)
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=colormapping.get(category),
            label=category,
            alpha=alpha,
            edgecolor='black',
            s=120,
            marker='o' #marker
        )
    
    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in colormapping.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    ax.axis("equal")







# stylistics functions
def get_spacy_attributes(token):
    # Save all token attributes in a list
    token_attributes = [
        token.i,
        token.text,
        token.lemma_,
        token.is_punct,
        token.is_stop,
        token.morph,
        token.pos_,
        token.tag_,
        token.dep_,
        token.head,
        token.head.i,
        token.ent_type_,
    ]

    return token_attributes


def create_spacy_df(doc_attributes: list) -> pd.DataFrame:
    df_attributes = pd.DataFrame(
        doc_attributes,
        columns=[
            "token_i",
            "token_text",
            "token_lemma_",
            "token_is_punct",
            "token_is_stop",
            "token_morph",
            "token_pos_",
            "token_tag_",
            "token_dep_",
            "token_head",
            "token_head_i",
            "token_ent_type_",
        ],
    )
    return df_attributes

def save_spacy_df(spacy_df, filename, out_dir) -> None:
    Path(f"{out_dir}/spacy_books/").mkdir(exist_ok=True)
    spacy_df.to_csv(f"{out_dir}/spacy_books/{filename}_spacy.csv")


def compressrat(sents: list[str]):
    """
    Calculates the GZIP compress ratio and BZIP compress ratio for the first 1500 sentences in a list of sentences
    """
    # skipping the first that are often title etc
    selection = sents[2:1502]
    asstring = " ".join(selection)  # making it a long string
    encoded = asstring.encode()  # encoding for the compression

    # GZIP
    g_compr = gzip.compress(encoded, compresslevel=9)
    gzipr = len(encoded) / len(g_compr)

    # BZIP
    b_compr = bz2.compress(encoded, compresslevel=9)
    bzipr = len(encoded) / len(b_compr)

    return gzipr, bzipr


def cal_entropy(base, log_n, transform_prob):
    entropy = 0
    for count in transform_prob.values():
        if count > 0:  # Avoid log of zero
            probability = count / sum(transform_prob.values())
            entropy -= probability * (log(probability, base) - log_n)
    return entropy

def text_entropy(words, base=2, asprob=True):
    total_len = len(words) - 1
    bigram_transform_prob = Counter()
    word_transform_prob = Counter()

    # Loop through each word and calculate the probabilities
    for i, word in enumerate(words):
        if i > 0:
            bigram_transform_prob[(words[i-1], word)] += 1
            word_transform_prob[word] += 1

    if asprob:
        return word_transform_prob, bigram_transform_prob

    log_n = log(total_len, base)
    bigram_entropy = cal_entropy(base, log_n, bigram_transform_prob)
    word_entropy = cal_entropy(base, log_n, word_transform_prob)

    return bigram_entropy / total_len, word_entropy / total_len