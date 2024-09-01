# %%
# we want to compare the embeddings and tf-idf

from utils import *

from scipy.sparse import csr_matrix

# %%
# get data
with open('data/tfidf_data/tfidf_dataset.json', 'r') as f:
    dat = json.load(f)

df_emb = pd.DataFrame.from_dict(dat)

# %%
# get the metadata sheet to get categories
meta = pd.read_excel('data/DANISH_CORPUS_METADATA.xlsx')
print(len(meta))

# merge w metadata
df = df_emb.merge(meta, on='FILENAME', how='left')
df = df.drop_duplicates(subset=['FILENAME'])
print(len(df))

# we set a unique id for books (differentiating between authors titles) that we can use for the dendrogram
# function to get the first word/first 10 characters of a title

# apply get_first_word function from utils
df['Unique_ID'] = df['AUTH_LAST_MODERN'] + '_' + df['TITLE'].apply(get_first_word)#df['TITLE'].str.split(' ').str[0] #+ df.groupby('AUTH_LAST').cumcount().astype(str)

# get some of the columns
df = df[['FILENAME', 'EMBEDDING', 'TF_IDF', 'CATEGORY', 'Unique_ID']]

# we want to make sure that nothing gets moved around and the order is the same after we do the matrices
df = df.sort_values(by='Unique_ID').reset_index(drop=True)

df.head()

# %%
# check nans
nans_per_column = df.isna().sum()
print(nans_per_column)
print(f"Number of rows in df: {df.shape[0]}")
# %%

# Get the cosine-similarity matrices based on each method (embedding and tf-idf)

# prepare data for plotting

# Make matrix for the embeddings
embeddings_matrix_emb = np.stack(df['EMBEDDING'].values)
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
matrix_tfidf = np.stack(df['TF_IDF'].values)
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
# we try to plot each matric individually

# get ids
unique_ids = df['CATEGORY'].values

# Plot the cosine distance matrix for embeddings
plt.figure(figsize=(20, 20))
sns.heatmap(cosine_dist_matrix_emb, xticklabels=unique_ids, yticklabels=unique_ids, cbar=False)
plt.title('Cosine Distance Matrix - Embeddings')
plt.show()

# Plot the cosine distance matrix for TF-IDF
plt.figure(figsize=(20, 20))
sns.heatmap(cosine_dist_matrix_tfidf, xticklabels=unique_ids, yticklabels=unique_ids, cbar=False)
plt.title('Cosine Distance Matrix - TF-IDF')
plt.show()

# %%
# now we want to inspect the difference between them

# subtract each cell from the matching cell (and visualize as a heatmap)
# get the difference matrix
diff_matrix = cosine_dist_matrix_emb - cosine_dist_matrix_tfidf
print(diff_matrix.shape)

# plot the difference
plt.figure(figsize=(35, 35))
sns.heatmap(diff_matrix, xticklabels=unique_ids, yticklabels=unique_ids, cbar=False, cmap='coolwarm')
plt.title('Difference Between Embeddings and TF-IDF Cosine Distance Matrices')
plt.show()

# and as a clustermap
plt.figure(figsize=(45, 45))
sns.clustermap(diff_matrix, xticklabels=unique_ids, yticklabels=unique_ids, cmap='coolwarm', figsize=(45, 45))
plt.title('Difference Between Embeddings and TF-IDF Cosine Distance Matrices')
plt.show()


# %%
# PART 2: Eyeballing, dendrogram visualization

# we want to plot the dendrogram for both the whole and the subset

# make subset
# we filter out the rest and only see clustering of canon/historicals
subset = df.loc[df['CATEGORY'] != 'O']
print(len(subset))

from importlib import reload
import utils
reload(utils)
from utils import *

# define colors for PCA
color_mapping = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}
# add a column for the color
df['CATEGORY_COLOR'] = df['CATEGORY'].map(color_mapping)

# define the embeddings and tf-idf columns to be plotted
matrices = [cosine_dist_matrix_emb, cosine_dist_matrix_tfidf, diff_matrix]
# labels for the matrices
labels = ['Embeddings', 'TF-IDF', 'Difference (Embeddings - TF-IDF)']

# for each type of embedding:
for i, matrix in enumerate(matrices):
    # plot dendrogram
    print(labels[i])
    sns.set_style('whitegrid')
    plot_dendrogram_different_embeddings(df, matrix, 'CATEGORY', 'CATEGORY', method_name=labels[i], l=26, h=5)
    #plot_dendrogram_different_embeddings(subset, 'CATEGORY', 'Unique_ID', v, l=26, h=5)



        
# %%

# same for subset

# Get the indices of the subset in the original DataFrame
subset_indices = subset.index

# Subset the distance matrices (mapping by index)
subset_cosine_dist_matrix_emb = cosine_dist_matrix_emb[np.ix_(subset_indices, subset_indices)]
subset_cosine_dist_matrix_tfidf = cosine_dist_matrix_tfidf[np.ix_(subset_indices, subset_indices)]
subset_diff_matrix = diff_matrix[np.ix_(subset_indices, subset_indices)]

subset_matrices = [subset_cosine_dist_matrix_emb, subset_cosine_dist_matrix_tfidf, subset_diff_matrix]
subset_labels = ['Embeddings (Subset)', 'TF-IDF (Subset)', 'Difference (Embeddings - TF-IDF) (Subset)']

# for each type of embedding:
for i, matrix in enumerate(subset_matrices):
    # plot dendrogram
    print(subset_labels[i])
    plot_dendrogram_different_embeddings(subset, matrix, 'CATEGORY', 'Unique_ID', method_name=subset_labels[i], l=26, h=5)

# make into a one-dimensional matrix and then just correlate it
# %%

# PCA
from importlib import reload
import utils
reload(utils)
from utils import *

# make copy
dt = df.copy()
dt['TF_IDF'] = dt['TF_IDF'].apply(lambda x: np.array(x, dtype=np.float64))
dt['TF_IDF'] = dt['TF_IDF'].apply(lambda x: np.nan_to_num(x, nan=0.0))
subset_dt = dt.loc[dt['CATEGORY'] != 'O']

# the cols for PCA
data = ['EMBEDDING', 'TF_IDF']

for method in data:
        # plot PCA

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plot PCA for the entire dataset
    plot_pca_different_embeddings(axs[0], dt, method, f"Entire corpus, method: {method}", color_mapping)
    # Plot PCA for the subset
    plot_pca_different_embeddings(axs[1], subset_dt, method, f"Canon and historical novels ($n$={len(subset)}), method: {method}", color_mapping)
    axs[1].legend().remove()
    axs[1].set_ylabel("")

    # reduce fontsize label
    axs[0].legend(fontsize=9, loc='lower right')

    plt.tight_layout()

    plt.show()


# %%
