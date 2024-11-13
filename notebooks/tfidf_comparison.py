# %%
# datasets
from datasets import load_dataset
import pandas as pd
import re

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# analysis
import numpy as np

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage
from scipy.sparse import csr_matrix

# %%
# we get our metadata (huggingface dataset)
ds = load_dataset("chcaa/memo-canonical-novels")
meta = ds["train"].to_pandas()
meta.columns = [x.lower() for x in meta.columns]
meta.head()
# %%
# get embeddings
df_emb = pd.read_json('data/meanpool__intfloat__multilingual-e5-large-instruct_identify_author.json', lines=True)
df_emb['embedding'] = df_emb['embedding'].apply(lambda x: np.array(x, dtype=np.float64)) # make sure the embeddings are np arrays
df_emb.head()
# %%
# merge w metadata & clean
df = df_emb.merge(meta, on='filename', how='left')
df = df.drop_duplicates(subset=['filename'])
print(len(df))

# we set a unique id for books (differentiating between authors titles)
# function to get the first word/first 10 characters of a title
def get_first_word(title):
    words = str(title).split(' ')
    if len(words) > 0 and len(words[0]) <= 10:
        return words[0]
    return words[0][:10]

# apply it
df['unique_id'] = df['auth_last_modern'] + '_' + df['title'].apply(get_first_word)

# we drop the row w nan in year column
df = df.dropna(subset=['year'])
print(len(df))
df.head()

# %%
# clean the texts
file_dict = dict(zip(df['filename'], df['text']))
print('length of file_dict:', len(file_dict))

# Process the texts:
for filename in file_dict:
    # replace all line breaks with space
    file_dict[filename] = file_dict[filename].replace('\n', ' ')
    # and lowercase everything
    file_dict[filename] = file_dict[filename].lower()
    # and remove all punctuation
    file_dict[filename] = re.sub(r'[^\w\s]', '', file_dict[filename])
    # remove underscores
    file_dict[filename] = file_dict[filename].replace('_', ' ')
    # and remove all double spaces
    file_dict[filename] = file_dict[filename].replace('  ', ' ')
    # and remove all numbers
    file_dict[filename] = re.sub(r'\d+', '', file_dict[filename])

# %%
# now we do tf-idf

# create the vectorizer
vectorizer = TfidfVectorizer()
# fit the vectorizer
X = vectorizer.fit_transform(file_dict.values())
# get the feature names
feature_names = vectorizer.get_feature_names_out()
# get the tf-idf values
tfidf = X.toarray()

# %%
# make this a df with the filenames as index
tfidf_df = pd.DataFrame(tfidf, columns=feature_names, index=file_dict.keys())
tfidf_df.head()
# %%
# we want to get rid of some of the "trash", like the features that were just OCR errors and such
# this is a number gotten from inspection of the matrix
threshold = 170
cols_to_keep = list(tfidf_df.columns)[:-threshold]
tfidf_df = tfidf_df[cols_to_keep].copy()
print(tfidf_df.shape)

# we want to know how many nan values there are (and why)
nans_per_column = tfidf_df.isna().sum()
print(nans_per_column)

# %%
# make the filename column normalised so we can merge
#tfidf_df['FILENAME'] = tfidf_df.index.str.replace('.txt', '')
tfidf_df = tfidf_df.reset_index(drop=True)
tfidf_df.head()
# %%
# merge it with the metadata
df = df[['filename', 'embedding', 'unique_id', 'category']].copy()
df = df.merge(tfidf_df, left_index=True, right_index=True, how='left')
df.head()

# %%
# we need to make the tf-idf values into a list (so theyre a vector in one column)
tfidf_columns = list(df.columns)[4:] # choose all cols (tfidf cols) except the first two (filename, embedding, unique_id)
df['tf_idf']= df[tfidf_columns].values.tolist()
# drop all the tf-idf columns
df = df[['filename', 'embedding', 'tf_idf', 'unique_id', 'category']].copy()
df.head()

# %%

# ANALYSIS: Compare the embeddings with the tf-idf values
df = df.sort_values(by='unique_id').reset_index(drop=True)

# prepare data for plotting
# Make matrix for the embeddings
embeddings_matrix_emb = np.stack(df['embedding'].values)
# get cosine distances
cosine_dist_matrix_emb = cosine_distances(embeddings_matrix_emb)

# raise error for not symmetric
if cosine_dist_matrix_emb.shape[0] != cosine_dist_matrix_emb.shape[1]:
    raise ValueError("Distance matrix is not square.")

print(cosine_dist_matrix_emb.shape)
print(cosine_dist_matrix_emb[:5, :5])

# Linkage for the dendrogram
Z_emb = linkage(cosine_dist_matrix_emb, method='ward')

# %%

# Make matrix for the tf-idf (takes a bit longer)
matrix_tfidf = np.stack(df['tf_idf'].values)
matrix_tfidf = np.array(matrix_tfidf, dtype=np.float64) # gotta make sure it's float64 (otherwise I get unsupported data types in input warning)

# make it a sparse matrix instead
matrix_tfidf_sparse = csr_matrix(matrix_tfidf)
print(matrix_tfidf.dtype)

# check on nans
nans_before = np.isnan(matrix_tfidf_sparse.data).sum()
total_elements = matrix_tfidf_sparse.data.size
print(f"Number of NaNs replaced: {nans_before}")
print(f"Total number of elements: {total_elements}")
# print the percentage of nans
print(f"Percentage of NaNs: {nans_before/total_elements*100:.2f}%")
print(nans_before)

# fillna with 0
matrix_tfidf_sparse.data = np.nan_to_num(matrix_tfidf_sparse.data, nan=0.0)  # otherwise I get contains nan warning

# make matrix
cosine_dist_matrix_tfidf = cosine_distances(matrix_tfidf_sparse) #(matrix_tfidf)

print(cosine_dist_matrix_tfidf.shape)
print(cosine_dist_matrix_tfidf[:5, :5])

if cosine_dist_matrix_tfidf.shape[0] != cosine_dist_matrix_tfidf.shape[1]:
    raise ValueError("Distance matrix is not square.")

# Linkage for the dendrogram
Z_tfidf = linkage(cosine_dist_matrix_tfidf, method='ward')
# %%

# get ids
categories = df['category'].values
# now we want to inspect the difference between cosine distance matrices

# subtract each cell from the matching cell (and visualize as a heatmap)
# get the difference matrix
diff_matrix = cosine_dist_matrix_emb - cosine_dist_matrix_tfidf
print(diff_matrix.shape)

# plot the difference
plt.figure(figsize=(35, 35))
sns.heatmap(diff_matrix, xticklabels=categories, yticklabels=categories, cbar=False, cmap='coolwarm')
plt.title('Difference Between Embeddings and TF-IDF Cosine Distance Matrices')
plt.show()

# and as a clustermap
plt.figure(figsize=(45, 45))
sns.clustermap(diff_matrix, xticklabels=categories, yticklabels=categories, cmap='coolwarm', figsize=(45, 45))
plt.title('Difference Between Embeddings and TF-IDF Cosine Distance Matrices')
plt.show()

# %%

# PART 2: Eyeballing, dendrogram visualization
from utils import plot_dendrogram_different_embeddings

# we want to plot the dendrogram for both the whole and the subset

# make subset: we filter out the rest and only see clustering of canon/historicals
subset = df.loc[df['category'] != 'O']
print('len of canon/historical subset', len(subset))

# define colors for PCA
color_mapping = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}
# add a column for the color
df['color_cat'] = df['category'].map(color_mapping)

# define the embeddings and tf-idf columns to be plotted
matrices = [cosine_dist_matrix_emb, cosine_dist_matrix_tfidf, diff_matrix]
# labels for the matrices
labels = ['Embeddings', 'TF-IDF', 'Difference (Embeddings - TF-IDF)']

# for each type of embedding:
for i, matrix in enumerate(matrices):
    # plot dendrogram
    print(labels[i])
    sns.set_style('whitegrid')
    plot_dendrogram_different_embeddings(df, matrix, 'category', 'category', method_name=labels[i], l=26, h=5)

# # to plot the dendrograms using only the subset and the actual title names (unique ids), uncomment below
# subset_indices = subset.index
# # Subset the distance matrices (mapping by index)
# subset_cosine_dist_matrix_emb = cosine_dist_matrix_emb[np.ix_(subset_indices, subset_indices)]
# subset_cosine_dist_matrix_tfidf = cosine_dist_matrix_tfidf[np.ix_(subset_indices, subset_indices)]
# subset_diff_matrix = diff_matrix[np.ix_(subset_indices, subset_indices)]
# subset_matrices = [subset_cosine_dist_matrix_emb, subset_cosine_dist_matrix_tfidf, subset_diff_matrix]
# subset_labels = ['Embeddings (Subset)', '', 'Difference (Embeddings - TF-IDF) (Subset)']

# for i, matrix in enumerate(subset_matrices):
#     # plot dendrogram
#     print(subset_labels[i])
#     sns.set_style('whitegrid')
#     plot_dendrogram_different_embeddings(subset, matrix, 'category', 'unique_id', method_name=subset_labels[i], l=26, h=5)


# %%
# plot pca of itfidf
from utils import plot_pca_different_embeddings

df['tf_idf'] = df['tf_idf'].apply(lambda x: np.array(x, dtype=np.float64))
df['tf_idf'] = df['tf_idf'].apply(lambda x: np.nan_to_num(x, nan=0.0))

color_mapping = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

print('Plotting PCA of TF-IDF values')
fig, axs = plt.subplots(1, 2, figsize=(9, 4))

# Plot PCA for the entire dataset
plot_pca_different_embeddings(axs[0], df, 'tf_idf', f"Entire corpus", color_mapping)
# Plot PCA for the subset
plot_pca_different_embeddings(axs[1], subset, 'tf_idf', f"Canon and historical novels", color_mapping)

# layout
axs[1].legend().remove()
axs[1].set_ylabel("")
axs[0].legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.show()
# %%
