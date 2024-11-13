# %% 
#!pip install -r requirements.txt

# %%
# datasets
from datasets import Dataset
from datasets import load_dataset
import pandas as pd

# plotting
from utils import plot_dendrogram, plot_pca
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import umap

# analysis
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA



# %%
# we get the saved embeddings (model used in paper)
df_emb = pd.read_json('data/meanpool__intfloat__multilingual-e5-large-instruct_identify_author.json', lines=True)
df_emb['embedding'] = df_emb['embedding'].apply(lambda x: np.array(x, dtype=np.float64)) # make sure the embeddings are np arrays
df_emb.head()

# %%
# Alternatively, you have created new embeddings – then set the path according to which were created
# if using an unpooled version, we pool in below cell (if raw then pool)
emb_path = None # set to something like 'interim/meanpool__intfloat__multilingual-e5-large-instruct_identify_author'
data_path = 'data/'

if emb_path is not None:
    ds = Dataset.load_from_disk(data_path+emb_path)
    df_emb = ds.to_pandas()
    print(len(df_emb))

    # if not using pooled embeddings, we pool them here
    # this takes some time
    if 'meanpool' not in emb_path:

        def mean_pooling(dataset: Dataset):
            out = []
            for novel in tqdm(dataset):
                chunk_embs = novel["embedding"]
                emb = np.mean(chunk_embs, axis=0)
                out.append(emb)
            return out
        
        df_emb['embedding'] = mean_pooling(ds)

    # rename columns, drop raw embeddings & text chunks
    df_emb.rename(columns={'novel': 'filename'}, inplace=True)
    df_emb = df_emb[['filename', 'embedding']].copy()
    df_emb.head()


# %%
# we get our metadata (huggingface dataset)
ds = load_dataset("chcaa/memo-canonical-novels")
meta = ds["train"].to_pandas()
meta.columns = [x.lower() for x in meta.columns]
meta.head()
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

#%%
emb_matrix = np.stack(df['embedding'].values)
cosine_matrix = cosine_similarity(emb_matrix)
# # plot the cosim in a heatmap (uncomment to plot)
# sns.heatmap(cosine_matrix, annot=True, xticklabels=df['unique_id'], yticklabels=df['unique_id'])

# get the range of cosim
print(f"Cosim, Min: {cosine_matrix.min()}, Max: {cosine_matrix.max()}")

# # plot histogram of cosims (uncomment to plot)
# # drop the diagonal
# cosine_matrix = cosine_matrix[~np.eye(cosine_matrix.shape[0], dtype=bool)].reshape(cosine_matrix.shape[0], -1)
# sns.set_style('whitegrid')
# plt.figure(figsize=(10, 5))
# sns.histplot(cosine_matrix.flatten(), kde=True)
# plt.title("Cosine similarity distribution of pooled embeddings")
# plt.show()

# %%
# let's see clustering of all books as well as for canon/historicals
# make subset
subset = df.loc[df['category'] != 'O']
print(len(subset))

# plot (both subset and full) dendrograms
plot_dendrogram(df, 'category', 'category', l=26, h=5)
plot_dendrogram(subset, 'category', 'unique_id', l=26, h=5)


# %%

# PCA
# get colors
color_mapping = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
# Plot PCA for the entire dataset
plot_pca(axs[0], df, "Entire corpus", color_mapping)
# Plot PCA for the subset
plot_pca(axs[1], subset, f"Canon and historical novels ($n$={len(subset)})", color_mapping)
# layout
axs[1].legend().remove()
axs[1].set_ylabel("")
axs[0].legend(fontsize=8, loc='lower left')
axs[0].set_title("Entire corpus", fontsize=13)
axs[1].set_title("Canon and historical novels", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# PCA 2
# # subsets of the first and last 15 years
sorted_df_year = df.sort_values(by='year', ascending=True)
# get the first 419 in a df and the last in another
df_first_15 = sorted_df_year.iloc[:419]
df_last_15 = sorted_df_year.iloc[419:]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Plot PCA for the entire dataset
plot_pca(axs[0], df_first_15, f"{df_first_15['year'].min()}-{df_first_15['year'].max()}, ($n$={len(df_first_15)})", color_mapping)
# Plot PCA for the subset
plot_pca(axs[1], df_last_15, f"{df_last_15['year'].min()}-{df_last_15['year'].max()} ($n$={len(df_last_15)})", color_mapping)
# layout
axs[1].legend().remove()
axs[1].set_ylabel("")
axs[0].legend(fontsize=9, loc='lower left')
# # # set x and y lims manually to the same scale
xlims = [-0.075, 0.1]
ylims = [-0.1, 0.075]
axs[0].set_xlim(xlims)
axs[0].set_ylim(ylims)
axs[1].set_xlim(xlims)
axs[1].set_ylim(ylims)
axs[0].set_title(f"{int(df_first_15['year'].min())}-{int(df_first_15['year'].max())}, ($n$={len(df_first_15)})", fontsize=13)
axs[1].set_title(f"{int(df_last_15['year'].min())}-{int(df_last_15['year'].max())}, ($n$={len(df_last_15)})", fontsize=13)
plt.tight_layout()
plt.show()


# %%
# PART II: we want to know if our groups are becoming more similar over time

# we want to find put whether the historical and canon novels have become more similar over time in terms of embeddings
# we will use the cosine similarity for this

# we do this naively at first, without up or downsampling

# make sure years are numbers
df['year'] = df['year'].astype(int)

# get a df with just the rows in groups
hist_and_canon = df.loc[df['category'].isin(['HISTORICAL', 'CANON_HISTORICAL', 'LEX_CANON', 'Ce_canon'])]

## Start a loop over the years
mean_similarity_dict = {}

# window size and step size
window_size = 5
step_size = 1

# set sampling
sampling = False
sample_size = 4

# increase if we want to do more runs
number_of_runs = 1

# Get the minimum and maximum years in the dataset
min_year = df['year'].min()
max_year = df['year'].max()

runs_list = []

for run in range(number_of_runs):

    # Loop over the range of years with a rolling window
    for start_year in range(min_year, max_year - window_size + 1, step_size):

        temp = {}

        # Define rolling window range for each window
        year_range = list(range(start_year, start_year + window_size))
        range_label = f"{year_range[0]}-{year_range[-1]}"

        # PART I: get the INTRAGROUP similarity
        # make groups
        # for each group
        # get the embeddings for the current window and the cosine similarity of the current window

        # making the groups w sampling
        # HIST & CANON
        # get what is in the window and os not 'O'
        subset = df.loc[(df['year'].isin(year_range)) & (df['category'] != 'O')]
        # HISTORICAL
        historical = df.loc[(df['year'].isin(year_range)) & (df['category'].isin(['HISTORICAL', 'CANON_HISTORICAL']))]
        # CANON
        e_canon = df.loc[(df['year'].isin(year_range)) & (df['category'].isin(['LEX_CANON', 'Ce_canon', 'CANON_HISTORICAL']))]
        # OTHER
        # get whats in the range and is 'O'
        others = df.loc[(df['year'].isin(year_range)) & (df['category'] == 'O')]
        # NONCANON
        non_canon = df.loc[(df['year'].isin(year_range)) & (~df['category'].isin(['LEX_CANON', 'Ce_canon', 'CANON_HISTORICAL']))]
        # TOTAL
        df_total = df.loc[df['year'].isin(year_range)]

        group_eb = {'subset': subset, 'historical': historical, 'e_canon': e_canon,
                            'other': others, 'non_canon': non_canon, 'df_total': df_total}

        if sampling == True:
            # sample from the groups

            for key in group_eb:
                group = group_eb[key]
                group_eb[key] = group.sample(sample_size, random_state=run) if len(group) > sample_size else group

        # function to get mean & stds of cosim in group
        def get_cosim_mean_std(key):
            data = group_eb[key]
            embeddings = np.stack(data['embedding'].values)
            mean_cosim = cosine_similarity(embeddings).mean()
            std_cosim = cosine_similarity(embeddings).std()
            return mean_cosim, std_cosim

        # HIST and CANON
        # get subset embeddings
        subset_mean, subset_std = get_cosim_mean_std('subset')
        temp['HIST_CANON_COSIM_MEAN'] = subset_mean #subset_cosine_similarity
        temp['HIST_CANON_COSIM_STD'] = subset_std #subset_cosine_similarity_std

        # HIST
        # get historical embeddings
        hist_mean, hist_std = get_cosim_mean_std('historical')
        temp['HIST_COSIM_MEAN'] = hist_mean
        temp['HIST_COSIM_STD'] = hist_std

        # CANON
        # get canon embeddings
        c_mean, c_std = get_cosim_mean_std('e_canon')
        temp['CANON_COSIM_MEAN'] = c_mean
        temp['CANON_COSIM_STD'] = c_std

        # OTHER
        # get other embeddings
        o_mean, o_std = get_cosim_mean_std('other')
        temp['OTHERS_COSIM_MEAN'] = o_mean
        temp['OTHERS_COSIM_STD'] = o_std

        # NON-CANON
        # get non-canon embeddings
        nc_mean, nc_std = get_cosim_mean_std('non_canon')
        temp['NONCANON_COSIM_MEAN'] = nc_mean
        temp['NONCANON_COSIM_STD'] = nc_std

        # TOTAL
        # get total embeddings
        t_mean, t_std = get_cosim_mean_std('df_total')
        temp['TOTAL_COSIM_MEAN'] = t_mean
        temp['TOTAL_COSIM_STD'] = t_std

        temp['N_BOOKS'] = [len(df_total), len(others), len(subset), len(historical), len(e_canon)]

        # PART II: get INTERGROUP similarity, i.e., between groups, e.g. canon vs non-canon
        # we want to see if the difference btw canon and noncanon decreases over time
        # we get the average embedding vector for the window for both non-canon and canon
        # and then get the cosine similarity between those average embeddings

        # get the mean embeddings of the current window for each group
        historical_mean = group_eb['historical']['embedding'].mean(axis=0)
        e_canon_mean = group_eb['e_canon']['embedding'].mean(axis=0)
        others_mean = group_eb['other']['embedding'].mean(axis=0)
        non_canon_mean = group_eb['non_canon']['embedding'].mean(axis=0)

        # get the cosine similarity btw canon and non-canon
        canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, e_canon_mean])).mean()
        temp['CANON_NONCANON_COSIM'] = canon_noncanon_similarity

        # get the cosine similarity btw historicals and non-canon
        hist_noncanon_similarity = cosine_similarity(np.stack([historical_mean, others_mean])).mean()
        temp['HIST_OTHERS_COSIM'] = hist_noncanon_similarity

        # get the cosine similarity btw historicals and canon
        hist_canon_similarity = cosine_similarity(np.stack([historical_mean, e_canon_mean])).mean()
        temp['HIST_CANON_COSIM_MEAN'] = hist_canon_similarity

        # save
        mean_similarity_dict[range_label] = temp
        #historical_cosine_similarity, e_canon_cosine_similarity, subset_cosine_similarity, others_similarity, non_canon_similarity, total_similarity, canon_noncanon_similarity, hist_noncanon_similarity, [len(df_total), len(others), len(subset), len(historical), len(e_canon)]

    # done

    # Format df
    sim_df = pd.DataFrame.from_dict(mean_similarity_dict, orient='index').reset_index()
    sim_df = sim_df.rename(columns={"index": "year_RANGE"})

    sim_df['START_year'] = sim_df['year_RANGE'].apply(lambda x: int(x.split('-')[0]))

    # append df to list
    runs_list.append(sim_df)
    sim_df.head(20)


# plot 
# make 3 plots on a row with the similarities
sim_cols = ['CANON_COSIM_MEAN', 'NONCANON_COSIM_MEAN', 'TOTAL_COSIM_MEAN'] # 'HIST_CANON_COSIM_MEAN', 'HIST_COSIM_MEAN', 'OTHERS_COSIM_MEAN', 
colors = ['#75BCC6', '#BA5C12', 'grey', 'purple'] # 'orange', 'yellow',
labels = ['e_canon', 'HISTORICAL/CANON', 'NON_CANON', 'ALL']


correlation_vals = {col: [] for col in sim_cols}

fig, axs = plt.subplots(1, len(sim_cols), figsize=(23, 4), dpi=500)

for idx, frame in enumerate(runs_list):

    corrs = []

    for i, col in enumerate(sim_cols):
        axs[i].plot(frame['START_year'], frame[col], label=labels[i], color=colors[i], alpha=0.5, linewidth=1)
        axs[i].set_xlabel("t")
        axs[i].set_ylabel(r"$\overline{x}$ cosine similarity")
        #axs[i].legend()

        # also get correlation and make title
        corr, pval = spearmanr(frame['START_year'], frame[col])
        correlation_vals[col].append(corr)

        if pval < 0.01:
            axs[i].set_title(f"Spearman $r$ = {round(corr, 2)}**")
        elif pval < 0.05:
            axs[i].set_title(f"Spearman $r$ = {round(corr, 2)}*")
        else:
            axs[i].set_title(f"Spearman $r$ = {round(corr, 2)}")
        # drop ylabel for subsequent plots
        if i > 0:
            axs[i].set_ylabel("")
        
    #correlation_vals[f'run_{idx}'] = dict(zip(sim_cols, corrs))

for i, col in enumerate(sim_cols):
    avg_corr = np.mean(correlation_vals[col])  # Calculate the average correlation
    axs[i].set_title(f"{col.split('_')[0]}, Average Spearman $r$ = {round(avg_corr, 2)}", fontsize=12)

plt.show()

# plot c/nc & h/nc similarity

plt.figure(figsize=(7, 4))

corrs = []

for i, frame in enumerate(runs_list):

    plt.plot(frame['START_year'], frame['CANON_NONCANON_COSIM'], label='Canon vs non-canon similarity', color='#75BCC6', linewidth=2, 
             alpha=0.3)
    # get spearmanr
    corr, pval = spearmanr(frame['START_year'], frame['CANON_NONCANON_COSIM'])
    print(f"CORR = {round(corr, 2)}** {round(pval,2)}")

    corrs.append(corr)

    # get the correlation after the year
    year = 1875
    sim_df_after_1874 = frame.loc[frame['START_year'] >= year]
    corr_threshold, pval_threshold = spearmanr(sim_df_after_1874['START_year'], sim_df_after_1874['CANON_NONCANON_COSIM'])
    print(f"after {year}: {round(corr_threshold,2)}, p-value: {round(pval_threshold,2)}")

plt.title(f"Spearman $r$ = {round(corr, 2)}**")
plt.show()

# %%

# SANITY CHECK

# Now we do the same but keeping our data scarcity in mind, so simulating means, upsampling in a sense
# we want to do the same as above, but instead of sampling, we want to simulate a sitribution for each window

# we load the functions for simulation that can upsample either from a gaussian distribution or using bootstrap sampling
from utils import simulate_cosim_between_groups, simulate_cosim_mean_std

# %%
df['category'].value_counts()   
# %%
# SET PARAMETERS

# get a df with just the rows in groups
# hist_and_canon = df.loc[df['category'].isin(['HISTORICAL', 'CANON_HISTORICAL', 'LEX_CANON', 'Ce_canon'])]

# window size and step size
window_size = 4
step_size = 1
# Number of simulations
num_simulations = 1000

print(f'Window size: {window_size}, Step size: {step_size}, No. of simulations per window: {num_simulations}')

# type of simulation
sim_type = 'gaussian' # choice btw. 'gaussian' or 'bootstrap'
print(f'NB. type: {sim_type} simulation of data')

##

# Get the minimum and maximum years in the dataset
min_year = df['year'].min()
max_year = df['year'].max()

# to store results
mean_similarity_dict = {}

# Loop over the range of years with a rolling window
for start_year in range(min_year, max_year - window_size + 1, step_size):

    temp = {}

    # Define rolling window range for each window
    year_range = list(range(start_year, start_year + window_size))
    range_label = f"{year_range[0]}-{year_range[-1]}"

    # Subset the data for the current window
    subset = df.loc[(df['year'].isin(year_range)) & (df['category'] != 'O')]
    historical = df.loc[(df['year'].isin(year_range)) & (df['category'].isin(['HISTORICAL', 'CANON_HISTORICAL']))]
    e_canon = df.loc[(df['year'].isin(year_range)) & (df['category'].isin(['LEX_CANON', 'CE_CANON', 'CANON_HISTORICAL']))]
    others = df.loc[(df['year'].isin(year_range)) & (df['category'] == 'O')]
    non_canon = df.loc[(df['year'].isin(year_range)) & (~df['category'].isin(['LEX_CANON', 'CE_CANON', 'CANON_HISTORICAL']))]
    df_total = df.loc[df['year'].isin(year_range)]

    # make subset dictionary per window for easier handling
    # empty it
    group_eb = {}
    group_eb = {'subset': subset, 'historical': historical, 'e_canon': e_canon,
                        'other': others, 'non_canon': non_canon, 'df_total': df_total}

    # HIST and CANON
    subset_mean, subset_std = simulate_cosim_mean_std(group_eb, 'subset', num_simulations, sim_type)
    temp['HIST_CANON_COSIM_MEAN'] = subset_mean
    temp['HIST_CANON_COSIM_STD'] = subset_std

    # HIST
    hist_mean, hist_std = simulate_cosim_mean_std(group_eb, 'historical', num_simulations, sim_type)
    temp['HIST_COSIM_MEAN'] = hist_mean
    temp['HIST_COSIM_STD'] = hist_std

    # CANON
    c_mean, c_std = simulate_cosim_mean_std(group_eb, 'e_canon', num_simulations, sim_type)
    temp['CANON_COSIM_MEAN'] = c_mean
    temp['CANON_COSIM_STD'] = c_std

    # OTHER
    o_mean, o_std = simulate_cosim_mean_std(group_eb, 'other', num_simulations, sim_type)
    temp['OTHERS_COSIM_MEAN'] = o_mean
    temp['OTHERS_COSIM_STD'] = o_std

    # NON-CANON
    nc_mean, nc_std = simulate_cosim_mean_std(group_eb, 'non_canon', num_simulations, sim_type)
    temp['NONCANON_COSIM_MEAN'] = nc_mean
    temp['NONCANON_COSIM_STD'] = nc_std

    # TOTAL
    t_mean, t_std = simulate_cosim_mean_std(group_eb, 'df_total', num_simulations, sim_type)
    temp['TOTAL_COSIM_MEAN'] = t_mean
    temp['TOTAL_COSIM_STD'] = t_std

    temp['N_BOOKS'] = [len(df_total), len(others), len(subset), len(historical), len(e_canon)]

    # and
    # Intergroup similarity calculations
    mean_similarity_c_nc = simulate_cosim_between_groups(group_eb, 'e_canon', 'non_canon', num_simulations, sim_type)
    temp['CANON_NONCANON_COSIM'] = mean_similarity_c_nc

    mean_similarity_c_hist = simulate_cosim_between_groups(group_eb, 'e_canon', 'historical', num_simulations, sim_type)
    temp['HIST_CANON_COSIM'] = mean_similarity_c_hist

    # hist / noncanon cosim
    mean_similarity_hist_nc = simulate_cosim_between_groups(group_eb, 'historical', 'non_canon', num_simulations, sim_type)
    temp['HIST_NONCANON_COSIM'] = mean_similarity_hist_nc

    # Save the results
    mean_similarity_dict[range_label] = temp

# Format the final dataframe
simulate_df = pd.DataFrame.from_dict(mean_similarity_dict, orient='index').reset_index()
simulate_df = simulate_df.rename(columns={"index": "year_RANGE"})

simulate_df['START_year'] = simulate_df['year_RANGE'].apply(lambda x: int(x.split('-')[0]))

# print the info
print("number of simulations per group, per window,", num_simulations)

# make sure there was no group with less than 2 books in a window, the column is a list with group sizes
print(simulate_df['N_BOOKS'].apply(lambda x: min(x)).min(), "is the smallest group size in a window")

# %%

# plot 
# make 3 plots on a row with the similarities
sim_cols = ['CANON_COSIM_MEAN','TOTAL_COSIM_MEAN', 'NONCANON_COSIM_MEAN'] # 'HIST_CANON_COSIM_MEAN', 'HIST_COSIM_MEAN', 'OTHERS_COSIM_MEAN', 'NONCANON_COSIM_MEAN', 
colors = ['#75BCC6', 'grey', 'purple'] # 'orange', 'yellow','#BA5C12',
labels = ['e_canon',  'ALL', 'NON_CANON'] # 'HISTORICAL/CANON', 'NON_CANON',

correlation_vals = {col: [] for col in sim_cols}

fig, axs = plt.subplots(1, len(sim_cols), figsize=(15, 2.5), dpi=500)

corrs = []

for i, col in enumerate(sim_cols):
    axs[i].plot(simulate_df['START_year'], simulate_df[col], label=labels[i], color=colors[i], alpha=0.5, linewidth=3)
    axs[i].set_xlabel("t", fontsize=14)
    axs[i].set_ylabel(r"$\overline{x}$ cosine similarity", fontsize=14)
    #axs[i].legend()

    # also get correlation and make title
    corr, pval = spearmanr(simulate_df['START_year'], simulate_df[col])
    correlation_vals[col].append(corr)

    if pval < 0.01:
        axs[i].set_title(f"Spearman $r$ = {round(corr, 2)}**")
    elif pval < 0.05:
        axs[i].set_title(f"Spearman $r$ = {round(corr, 2)}*")
    else:
        axs[i].set_title(f"Spearman $r$ = {round(corr, 2)}")

    # bigger labels
    axs[i].tick_params(axis='both', which='major', labelsize=10)
    # drop ylabel for subsequent plots
    if i > 0:
        axs[i].set_ylabel("")

    plt.tight_layout()
        
for i, col in enumerate(sim_cols):
    avg_corr = np.mean(correlation_vals[col])  # Calculate the average correlation
    axs[i].set_title(f"{col.split('_')[0].title()}, $r$ = {round(avg_corr, 2)}", fontsize=14)

plt.show()


# plot c/nc & h/nc similarity

plt.figure(figsize=(8.5, 2.5), dpi=500)
plt.plot(simulate_df['START_year'], simulate_df['CANON_NONCANON_COSIM'], label='Canon vs non-canon similarity', color='#75BCC6', linewidth=4, 
            alpha=0.5)
# get spearmanr
corr, pval = spearmanr(simulate_df['START_year'], simulate_df['CANON_NONCANON_COSIM'])
print(f"CORR = {round(corr, 2)}** {round(pval,2)}")

# get the correlation after the year XX
year = 1875
sim_df_after_1874 = simulate_df.loc[simulate_df['START_year'] >= year]
corr_threshold, pval_threshold = spearmanr(sim_df_after_1874['START_year'], sim_df_after_1874['CANON_NONCANON_COSIM'])
print(f"after {year}: {round(corr_threshold,2)}, p-value: {round(pval_threshold,2)}")

# make x and y ticks bigger
plt.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.title(f"Canon vs Non-canon, $r$ = {round(corr, 2)}", fontsize=14)
plt.show()

# %%
# same for hist/noncanon
plt.figure(figsize=(7, 4))
plt.plot(simulate_df['START_year'], simulate_df['HIST_NONCANON_COSIM'], label='Historical vs canon similarity', color='#FE7F2D', linewidth=4, 
            alpha=0.5)
# get spearmanr
corr, pval = spearmanr(simulate_df['START_year'], simulate_df['HIST_NONCANON_COSIM'])
print(f"CORR = {round(corr, 2)}** {round(pval,2)}")
plt.title(f"Historical vs Non-canon, $r$ = {round(corr, 2)}")
plt.show()



# %%

# PART IV: we want to know the direction of similarity between canon and non-canon
# we try and gauge this by a simple pca (naive approach)

# make the extended color dict

sorted_df_year = df.sort_values(by='year', ascending=True)
print(len(df))

threshold = 419

dat = df[['embedding', 'category', 'year', 'e_canon']].copy()

sorted_df_year = dat.sort_values(by='year', ascending=True).reset_index(drop=True)

for i, row in sorted_df_year.iterrows():
    # make column
    if i < threshold and row['e_canon'] == 1:
        sorted_df_year.loc[i, 'cat_complex'] = 'C 1870-1888'
    elif i < threshold and row['e_canon'] != 1:
        sorted_df_year.loc[i, 'cat_complex'] = 'NC 1870-1888'
    else:
        if i>= threshold and row['e_canon'] == 1:
            sorted_df_year.loc[i, 'cat_complex'] = 'C 1888-1899'
        elif i >= threshold and row['e_canon'] != 1:
            sorted_df_year.loc[i, 'cat_complex'] = 'NC 1888-1899'

from utils import plot_pca_c_nc

# making colors
color_mapping = {'NC 1888-1899': '#129525', 'NC 1870-1888': '#A1E44D', 'C 1870-1888': '#7AFDD6', 'C 1888-1899': '#2A4C5E'}
color_mapping_means = {'NC 1888-1899_mean': '#129525', 'NC 1870-1988_mean': '#A1E44D', 'C 1870-1888_mean': '#7AFDD6', 'C 1888-1999_mean': '#2A4C5E'}

# apply
plt.figure(figsize=(8, 8))
plot_pca_c_nc(sorted_df_year, "First 15 years", colormapping=color_mapping, mean_colors=color_mapping_means)
plt.show()


# %%

# We want to do the exact same thing as above, but we want to plot a rolling window 
# we want to downsample majority group for each window, and get the mean embedding for each group

simple = df[['embedding', 'year', 'e_canon']].copy()
print(len(simple))

rolling_dict = {}

# window size and step size
window_size = 4
step_size = 1

# Get the minimum and maximum years in the dataset
min_year = df['year'].min()
max_year = df['year'].max()


# Loop over the range of years with a rolling window
for start_year in range(min_year, max_year - window_size + 1, step_size):

    temp = {}

    # Define rolling window range for each window
    year_range = list(range(start_year, start_year + window_size))
    range_label = f"{year_range[0]}-{year_range[-1]}"

    # Subset the data for the current window
    e_canon = simple.loc[(simple['year'].isin(year_range)) & (simple['e_canon'] == 1)]
    non_canon = simple.loc[(simple['year'].isin(year_range)) & (simple['e_canon'] != 1)]

    # # downsample the majority group
    # if len(e_canon) < len(non_canon):
    #     non_canon_sample = non_canon.sample(len(e_canon), random_state=42)
    # else:
    non_canon_sample = non_canon


    window_mean_canon = np.stack(e_canon['embedding'].values).mean(axis=0)
    window_mean_non_canon = np.stack(non_canon_sample['embedding'].values).mean(axis=0)
    
    rolling_dict[range_label] = {'e_canon_emb': e_canon['embedding'].values, 'non_canon_emb': non_canon_sample['embedding'].values, 'e_canon_mean': window_mean_canon, 'non_canon_mean': window_mean_non_canon, 'datalengths': [len(e_canon), len(non_canon_sample)]}

# check
print('no of windows:', len(rolling_dict))
pd.DataFrame.from_dict(rolling_dict, orient='index').head(20)


# Plot

# Generate distinguishable colors with wider spacing
num_windows = len(rolling_dict.keys())
blues = sns.color_palette("Blues", n_colors=num_windows, desat=1)
greens = sns.color_palette("Greens", n_colors=num_windows, desat=1)

# Color mapping for individual points
color_mapping_e_canon = {window: blues[i] for i, window in enumerate(rolling_dict.keys())}
color_mapping_non_canon = {window: greens[i] for i, window in enumerate(rolling_dict.keys())}

# Color mapping for mean points (extend the individual mappings)
color_mapping_e_canon_mean = {window: blues[i] for i, window in enumerate(rolling_dict.keys())}
color_mapping_non_canon_mean = {window: greens[i] for i, window in enumerate(rolling_dict.keys())}

# Fit PCA on all embeddings
all_embeddings = np.stack(simple['embedding'].values)
pca = PCA(n_components=2)
pca_fit = pca.fit(all_embeddings)

# Prepare the PCA results for plotting
pca_results = []


for key in rolling_dict.keys():
    e_canon_pca = pca_fit.transform(np.stack(rolling_dict[key]['e_canon_emb']))
    non_canon_pca = pca_fit.transform(np.stack(rolling_dict[key]['non_canon_emb']))
    e_canon_mean_pca = pca_fit.transform([rolling_dict[key]['e_canon_mean']])
    non_canon_mean_pca = pca_fit.transform([rolling_dict[key]['non_canon_mean']])

    
    # Append individual PCA results
    for i in range(len(e_canon_pca)):
        pca_results.append({
            'PCA1': e_canon_pca[i][0],
            'PCA2': e_canon_pca[i][1],
            'group': 'e_canon',
            'window': key,
            'color': color_mapping_e_canon[key],
        })
    
    for i in range(len(non_canon_pca)):
        pca_results.append({
            'PCA1': non_canon_pca[i][0],
            'PCA2': non_canon_pca[i][1],
            'group': 'non_canon',
            'window': key,
            'color': color_mapping_non_canon[key],
        })
    
    # Append mean PCA results (use the same color as the individual points)
    pca_results.append({
        'PCA1': e_canon_mean_pca[0][0],
        'PCA2': e_canon_mean_pca[0][1],
        'group': 'e_canon_mean',
        'window': key,
        'color': color_mapping_e_canon_mean[key],  # Use the same color as individual e_canon
    })
    
    pca_results.append({
        'PCA1': non_canon_mean_pca[0][0],
        'PCA2': non_canon_mean_pca[0][1],
        'group': 'non_canon_mean',
        'window': key,
        'color': color_mapping_non_canon_mean[key],  # Use the same color as individual non_canon
    })

# Convert the PCA results into a DataFrame for plotting
pca_df = pd.DataFrame(pca_results)

# Plotting
plt.figure(figsize=(8, 8))

# Scatter plot for individual embeddings and convex hulls
for window in pca_df['window'].unique():
    for group in ['e_canon', 'non_canon']:
        subset = pca_df[(pca_df['window'] == window) & (pca_df['group'] == group)]
        plt.scatter(
            subset['PCA1'], subset['PCA2'], 
            color=subset['color'].values[0], alpha=0.05, edgecolor='black', s=100, marker='x'
        )
        
        #   # uncomment to add convex hulls
        # if len(subset) > 2:  # Convex hull requires at least 3 points
        #     points = subset[['PCA1', 'PCA2']].values
        #     hull = ConvexHull(points)
        #     for simplex in hull.simplices:
        #         plt.plot(points[simplex, 0], points[simplex, 1], color=subset['color'].values[0], alpha=0.2, linewidth=2)

# Scatter plot for mean points
for group in ['e_canon_mean', 'non_canon_mean']:
    subset = pca_df[pca_df['group'] == group]
    plt.scatter(
        subset['PCA1'], subset['PCA2'],
        color=subset['color'].values, alpha=0.7, s=800, marker='o', edgecolor=subset['color'].values
    )

    # get labels
    color_mapping_two = {'C': '#75BCC6', 'NC': 'green'}
    legend_handles = [Patch(facecolor=color, label=label) for label, color in color_mapping_two.items()]
    plt.legend(handles=legend_handles, loc='upper right', fontsize=12)


plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.xlim(-0.03, 0.03)
plt.ylim(-0.03, 0.03)
plt.grid(True)
plt.show()


# %%

# try same w. umap

# same as before, making colors
num_windows = len(rolling_dict.keys())
blues = sns.color_palette("Blues", n_colors=num_windows, desat=1)
greens = sns.color_palette("Greens", n_colors=num_windows, desat=1)

color_mapping_e_canon = {window: blues[i] for i, window in enumerate(rolling_dict.keys())}
color_mapping_non_canon = {window: greens[i] for i, window in enumerate(rolling_dict.keys())}

# colors for mean points
color_mapping_e_canon_mean = {window: blues[i] for i, window in enumerate(rolling_dict.keys())}
color_mapping_non_canon_mean = {window: greens[i] for i, window in enumerate(rolling_dict.keys())}

all_embeddings = np.stack(simple['embedding'].values)
umap_fit = umap.UMAP(n_components=2).fit(all_embeddings)

umap_results = []

# fit umap on the groups
for key in rolling_dict.keys():
    e_canon_umap = umap_fit.transform(np.stack(rolling_dict[key]['e_canon_emb']))
    non_canon_umap = umap_fit.transform(np.stack(rolling_dict[key]['non_canon_emb']))
    e_canon_mean_umap = umap_fit.transform([rolling_dict[key]['e_canon_mean']])
    non_canon_mean_umap = umap_fit.transform([rolling_dict[key]['non_canon_mean']])

    # Append individual UMAP results
    for i in range(len(e_canon_umap)):
        umap_results.append({
            'UMAP1': e_canon_umap[i][0],
            'UMAP2': e_canon_umap[i][1],
            'group': 'e_canon',
            'window': key,
            'color': color_mapping_e_canon[key],
        })

    for i in range(len(non_canon_umap)):
        umap_results.append({
            'UMAP1': non_canon_umap[i][0],
            'UMAP2': non_canon_umap[i][1],
            'group': 'non_canon',
            'window': key,
            'color': color_mapping_non_canon[key],
        })

    # Append mean UMAP results (use the same color as the individual points)
    umap_results.append({
        'UMAP1': e_canon_mean_umap[0][0],
        'UMAP2': e_canon_mean_umap[0][1],
        'group': 'e_canon_mean',
        'window': key,
        'color': color_mapping_e_canon_mean[key],
    })

    umap_results.append({
        'UMAP1': non_canon_mean_umap[0][0],
        'UMAP2': non_canon_mean_umap[0][1],
        'group': 'non_canon_mean',
        'window': key,
        'color': color_mapping_non_canon_mean[key],
    })

# Convert the UMAP results into a DataFrame for plotting
umap_df = pd.DataFrame(umap_results)

# Plotting
plt.figure(figsize=(25, 20))

# plot for individual points
for window in umap_df['window'].unique():
    for group in ['e_canon', 'non_canon']:
        subset = umap_df[(umap_df['window'] == window) & (umap_df['group'] == group)]
        plt.scatter(
            subset['UMAP1'], subset['UMAP2'],
            color=subset['color'].values[0], alpha=0.1, edgecolor='black', s=100
        )

# plot mean points
for group in ['e_canon_mean', 'non_canon_mean']:
    subset = umap_df[umap_df['group'] == group]
    plt.scatter(
        subset['UMAP1'], subset['UMAP2'],
        color=subset['color'].values, alpha=0.5, edgecolor=subset['color'].values, s=3000, marker='o'
    )

# Set labels and show plot
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True)
plt.show()
# %%
