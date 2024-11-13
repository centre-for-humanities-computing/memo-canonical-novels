import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_distances
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from collections import Counter
from math import log
from sklearn.utils import resample


##
# plotting functions

# function to make dendrogram from embeddings
def plot_dendrogram(df, col_to_color, col_to_label, l, h, palette='Set2'):
    unique_categories = df[col_to_color].unique()

    # colors
    cat_map = dict(zip(df[col_to_label],df[col_to_color]))
    color_dict = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

    # prepare data for plotting
    embeddings_matrix = np.stack(df['embedding'].values)
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
    
    # update labels in used_colors, make titlecase
    used_colors = {k.replace('_', ' ').title(): v for k, v in used_colors.items()}
    # make "other" if O in used_colors
    if 'O' in used_colors:
        used_colors['Other'] = used_colors.pop('O')
    
    # layout
    plt.xlabel("Cosine Distance")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in used_colors.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    plt.show()


# PCA plot

def plot_pca(ax, data, title, colormapping):
    # Handle embeddings
    embeddings_array = np.array(data["embedding"].to_list(), dtype=np.float32)

    # Make labels titlecase
    colormapping = {k.replace('_', ' ').title(): v for k, v in colormapping.items()}
    # Replace 'o' with 'Other'
    colormapping['Other'] = colormapping.pop('O')

    
    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["category"] = data["category"].values
    df_pca["category"] = df_pca["category"].apply(lambda x: x.replace('_', ' ').title())
    # replace 'O' with 'Other'
    df_pca["category"] = df_pca["category"].apply(lambda x: 'Other' if x == 'O' else x)


    # We're gonna set a different alpha for the 'O' category
    alpha_dict = dict(zip(colormapping.keys(), [0.65 if x != 'Other' else 0.2 for x in colormapping.keys()]))
    # Update color dict to have titlecase

    # Plot each category
    for category in df_pca["category"].unique():
        subset = df_pca[df_pca["category"] == category]

        #marker = markers_dict.get(category) 
        alpha = alpha_dict.get(category)
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=colormapping.get(category),
            label=category,
            alpha=alpha,
            edgecolor='black',
            s=110,
            marker='o' #marker
        )

    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in colormapping.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    ax.axis("equal")


# making functions for the comparing types of embeddings
# we need to pass something else than df['embeddingS'] - so that is the only difference from the ones above

# function to make dendrogram from embeddings
def plot_dendrogram_different_embeddings(df, cosine_matrix, col_to_color, col_to_label, l, h, method_name=None, palette='Set2'):
    unique_categories = df[col_to_color].unique()

    # colors
    cat_map = dict(zip(df[col_to_label],df[col_to_color]))
    color_dict = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

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
    # update labels in used_colors, make titlecase
    used_colors = {k.replace('_', ' ').title(): v for k, v in used_colors.items()}
    # make "other" if O in used_colors
    if 'O' in used_colors:
        used_colors['Other'] = used_colors.pop('O')

    # layout
    plt.xlabel("Cosine Distance")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in used_colors.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    #plt.title(f'Canon/Historical tags in {len(df)} novels, label: {method_name}')
    plt.show()




def plot_pca_different_embeddings(ax, data, data_col, title, colormapping):
    # Handle embeddings
    embeddings_array = np.array(data[data_col].to_list(), dtype=np.float32)
    
    # Make labels titlecase
    colormapping = {k.replace('_', ' ').title(): v for k, v in colormapping.items()}
    # Replace 'o' with 'Other'
    colormapping['Other'] = colormapping.pop('O')

    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["category"] = data["category"].values
    df_pca["category"] = df_pca["category"].apply(lambda x: x.replace('_', ' ').title())
    # replace 'O' with 'Other'
    df_pca["category"] = df_pca["category"].apply(lambda x: 'Other' if x == 'O' else x)

    # We're gonna set a different alpha for the 'O' category
    alpha_dict = dict(zip(colormapping.keys(), [0.65 if x != 'Other' else 0.2 for x in colormapping.keys()]))

    # Plot each category
    for category in df_pca["category"].unique():
        subset = df_pca[df_pca["category"] == category]

        #marker = markers_dict.get(category) 
        alpha = alpha_dict.get(category)
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=colormapping.get(category),
            label=category,
            alpha=alpha,
            edgecolor='black',
            s=110,
            marker='o' #marker
        )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in colormapping.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    ax.axis("equal")


# simulation functions

# INTRAGROUP

def simulate_cosim_mean_std(group_df, key, num_simulations, type='gaussian'):
        data = group_df[key]
        if len(data) > 1:

            embeddings = np.stack(data['embedding'].values)
            cosim_matrix = cosine_similarity(embeddings)

            # exclude the diagonal of the matrix
            cosim_values = cosim_matrix[np.triu_indices_from(cosim_matrix, k=1)]

            if type == 'gaussian':
                simulated_means = []
                for _ in range(num_simulations):
                    # Simulate data from a normal distribution based on the sample mean and std
                    simulated_data = np.random.normal(cosim_values.mean(), cosim_values.std(), len(cosim_values))
                    simulated_means.append(simulated_data.mean())

                return np.mean(simulated_means), np.std(simulated_means)
            
            elif type == 'bootstrap':
                bootstrap_means = []
                for _ in range(num_simulations):
                    # Generate a bootstrap sample with replacement
                    bootstrap_sample = resample(cosim_values)
                    # Calculate the mean of the bootstrap sample
                    bootstrap_mean = np.mean(bootstrap_sample)

                    bootstrap_means.append(bootstrap_mean)

                return np.mean(bootstrap_means), np.std(bootstrap_means)
            
            else:
                print('simulation type not defined')
        else:
            return np.nan, np.nan
    
        
# INTERGROUP
def simulate_cosim_between_groups(df, key1, key2, num_simulations, type='gaussian'):
    # Get embeddings for each group
    data1 = df[key1]
    data2 = df[key2]
    
    if len(data1) > 1 and len(data2) > 1:
        embeddings1 = np.stack(data1['embedding'].values)
        embeddings2 = np.stack(data2['embedding'].values)

        mean_emb1_list = []
        mean_emb2_list = []

        if type == 'gaussian':
            for _ in range(num_simulations):
                # Simulate embeddings from normal distribution
                simulated_embedding1 = np.random.normal(embeddings1.mean(), embeddings1.std(), embeddings1.shape)
                simulated_embedding2 = np.random.normal(embeddings2.mean(), embeddings2.std(), embeddings2.shape)
                
                # Calculate mean embedding from each simulation
                mean_embedding1 = np.mean(simulated_embedding1, axis=0)
                mean_embedding2 = np.mean(simulated_embedding2, axis=0)

                # mean embedding 1 and 2
                mean_emb1_list.append(mean_embedding1)
                mean_emb2_list.append(mean_embedding2)
            
            # get overall mean of all simulation runs for each group
            # so i get two means
            overall_mean1 = np.mean(mean_emb1_list, axis=0)
            overall_mean2 = np.mean(mean_emb2_list, axis=0)

            # # Calculate cosine similarity between the mean embeddings
            similarity = cosine_similarity([overall_mean1], [overall_mean2])

            return similarity
        
        elif type == 'bootstrap':
            for _ in range(num_simulations):
                # Sample with replacement from both groups independently
                bootstrap_sample1 = resample(embeddings1)
                bootstrap_sample2 = resample(embeddings2)
                
                # Calculate mean embeddings for each bootstrap sample
                mean_embedding1 = np.mean(bootstrap_sample1, axis=0)
                mean_embedding2 = np.mean(bootstrap_sample2, axis=0)

                # mean embedding 1 and 2
                mean_emb1_list.append(mean_embedding1)
                mean_emb2_list.append(mean_embedding2)
            
            # get overall mean of all simulation runs for each group
            # so i get two means
            overall_mean1 = np.mean(mean_emb1_list, axis=0)
            overall_mean2 = np.mean(mean_emb2_list, axis=0)

            # # Calculate cosine similarity between the mean embeddings
            similarity = cosine_similarity([overall_mean1], [overall_mean2])


            return similarity
        
        else:
            print('simulation type not defined')
    else:
        return np.nan    




def plot_pca_c_nc(data, title, colormapping, mean_colors):

    # Handle embeddings
    new_rows = []

    for category in data["cat_complex"].unique():
        mean_embeddings = np.stack(data[data["cat_complex"] == category]["embedding"].to_list()).mean(axis=0)
        new_row = {"embedding": mean_embeddings, "cat_complex": category + '_mean'}
        new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)

    data = pd.concat([data, new_rows_df], ignore_index=True)

    embeddings_array = np.array(data["embedding"].to_list(), dtype=np.float32)
    
    # we fit the PCA on the embeddings
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["cat_complex"] = data["cat_complex"].values
    # Plot each category
    for category in df_pca["cat_complex"].unique():
        if category.endswith('mean') is False:
            subset = df_pca[df_pca["cat_complex"] == category]

            plt.scatter(
                subset["PCA1"],
                subset["PCA2"],
                color=colormapping.get(category),
                label=category,
                alpha=0.4,
                edgecolor='black',
                s=100,
                marker='o'
            )

    # plot mean points
    for category in df_pca["cat_complex"].unique():
        if category.endswith('mean') is True:
            subset = df_pca[df_pca["cat_complex"] == category]

            plt.scatter(
                subset["PCA1"],
                subset["PCA2"],
                color=mean_colors.get(category),
                label=category,
                alpha=0.9,
                edgecolor='black',
                s=300,
                marker='o'
            )

    legend_handles = [Patch(facecolor=color, label=label) for label, color in colormapping.items()]
    plt.legend(handles=legend_handles, loc='upper right')





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