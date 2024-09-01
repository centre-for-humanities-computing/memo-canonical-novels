# %% 
#!pip install -r requirements.txt
# %%
from importlib import reload
import utils
reload(utils)
from utils import *

# %%
# get data
path = 'data/meanpool__intfloat__multilingual-e5-large-instruct_identify_author'
from datasets import Dataset
ds = Dataset.load_from_disk(path)
df_emb = ds.to_pandas()
print(len(df_emb))
df_emb.head()

# %%
# do the processing (getting mean embddings if treating raw)
if path == 'emb_einstruct_author_raw':

    # count the number of lists in the embedding
    #df_emb['N_CHUNKS'] = df_emb['embedding'].apply(lambda x: len(x))

    def mean_pooling(dataset: Dataset):
        out = []
        for novel in tqdm(dataset):
            chunk_embs = novel["embedding"]
            emb = np.mean(chunk_embs, axis=0)
            out.append(emb)
        return out
    
    df_emb['EMBEDDING'] = mean_pooling(ds)

    # save this as a new dataset
    ds = Dataset.from_pandas(df_emb)
    ds.save_to_disk(f'{path.split('_')[0]}_processed')

    df_emb.head()

# %%
df_emb.columns = ['FILENAME', 'CHUNK', 'EMBEDDING_ORIGINAL', 'EMBEDDING']
df_emb = df_emb[['FILENAME', 'EMBEDDING']].copy()

df_emb.head()


# %%
# merge w metadata
meta = pd.read_excel('data/DANISH_CORPUS_METADATA_AUGUST20.xlsx')
print(len(meta))
meta.head()
df = df_emb.merge(meta, on='FILENAME', how='left')
df = df.drop_duplicates(subset=['FILENAME'])
print(len(df))

# we set a unique id for books (differentiating between authors titles)
# function to get the first word/first 10 characters of a title
def get_first_word(title):
    words = str(title).split(' ')
    if len(words) > 0 and len(words[0]) <= 10:
        return words[0]
    return words[0][:10]

# apply it
df['Unique_ID'] = df['AUTH_LAST_MODERN'] + '_' + df['TITLE'].apply(get_first_word)#df['TITLE'].str.split(' ').str[0] #+ df.groupby('AUTH_LAST').cumcount().astype(str)
df.head()
# %% make sure to drop the nan
df = df.dropna(subset=['YEAR'])
print(len(df))  

#%%
# plot the cosim in a heatmap
emb_matrix = np.stack(df['EMBEDDING'].values)
cosine_matrix = cosine_similarity(emb_matrix)
#sns.heatmap(cosine_matrix, annot=True, xticklabels=df['Unique_ID'], yticklabels=df['Unique_ID'])
cosine_matrix

# get the range of cosim
print(f"Min: {cosine_matrix.min()}, Max: {cosine_matrix.max()}")

# and get a distribution of the cosine similarities
# drop the diagonal
cosine_matrix = cosine_matrix[~np.eye(cosine_matrix.shape[0], dtype=bool)].reshape(cosine_matrix.shape[0], -1)

# plot
sns.set_style('whitegrid')
plt.figure(figsize=(10, 5))
sns.histplot(cosine_matrix.flatten(), kde=True)
plt.title("Cosine similarity distribution of pooled embeddings")
plt.show()

# %%
# let's try see clustering of all books as well as for canon/historicals
# make subset
subset = df.loc[df['CATEGORY'] != 'O']
print(len(subset))

# plot (both) dendrograms
plot_dendrogram(df, 'CATEGORY', 'CATEGORY', l=26, h=5)
plot_dendrogram(subset, 'CATEGORY', 'Unique_ID', l=26, h=5)


# %%

# try out PCA
# get colors
color_mapping = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

#color_mapping = dict(zip(df['CATEGORY'].unique(), colors[:len(df['CATEGORY'].unique())]))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Plot PCA for the entire dataset
plot_pca(axs[0], df, "Entire corpus", color_mapping)
# Plot PCA for the subset
plot_pca(axs[1], subset, f"Canon and historical novels ($n$={len(subset)})", color_mapping)
axs[1].legend().remove()
axs[1].set_ylabel("")

# reduce fontsize label
axs[0].legend(fontsize=8, loc='lower left')

plt.tight_layout()

plt.show()

# %%

# # subsets of the first and last 15 years
# df_first_15 = df.loc[df['YEAR'] <= 1885]
# df_last_15 = df.loc[df['YEAR'] > 1885]
color_mapping = {'O': '#129525', 'CE_CANON': '#356177', 'HISTORICAL': '#FE7F2D', 'CANON_HISTORICAL': '#9B7496', 'LEX_CANON': '#75BCC6'}#'#FCCA46'}

sorted_df_year = df.sort_values(by='YEAR', ascending=True)
print(len(df))

# gte the first 419 in a df and the last in another
df_first_15 = sorted_df_year.iloc[:419]
df_last_15 = sorted_df_year.iloc[419:]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Plot PCA for the entire dataset
plot_pca(axs[0], df_first_15, f"{df_first_15['YEAR'].min()}-{df_first_15['YEAR'].max()}, ($n$={len(df_first_15)})", color_mapping)
# Plot PCA for the subset
plot_pca(axs[1], df_last_15, f"{df_last_15['YEAR'].min()}-{df_last_15['YEAR'].max()} ($n$={len(df_last_15)})", color_mapping)
axs[1].legend().remove()
axs[1].set_ylabel("")

# reduce fontsize label
axs[0].legend(fontsize=9, loc='lower left')

# # # set x and y lims manually to the same scale
xlims = [-0.075, 0.1]

ylims = [-0.1, 0.075]
axs[0].set_xlim(xlims)
axs[0].set_ylim(ylims)
axs[1].set_xlim(xlims)
axs[1].set_ylim(ylims)


plt.tight_layout()


# %%


# Check of embeddings similarity

df.head()

a = df.loc[df['Unique_ID'] == 'Ewald_Makra'] 
b = df.loc[df['Unique_ID'] == 'Bang_Stille']

test = pd.concat([a,b])

embeddings_matrix = np.stack(test['EMBEDDING'].values)
cosine_dist_matrix = cosine_similarity(embeddings_matrix)
cosine_dist_matrix


# %%

# PART II: we want to know if our groups are becoming more similar over time

# we want to find put whether the historical and canon novels have become more similar over time in terms of embeddings
# we will use the cosine similarity for this

# make groups
historical = df.loc[df['CATEGORY'] == 'HISTORICAL']
e_canon = df.loc[df['E_CANON'] == 1]

# get the embeddings
historical_embeddings = np.stack(historical['EMBEDDING'].values)
e_canon_embeddings = np.stack(e_canon['EMBEDDING'].values)

# get the mean cosine similarity within the groups for each year
historical_cosine_similarities = []
e_canon_cosine_similarities = []
years = []

for year in range(1870, 1900):
    # get the embeddings for the year
    historical_year = historical.loc[historical['YEAR'] == year]
    e_canon_year = e_canon.loc[e_canon['YEAR'] == year]
    if len(historical_year) > 1 and len(e_canon_year) > 1:
        historical_embeddings_year = np.stack(historical_year['EMBEDDING'].values)
        e_canon_embeddings_year = np.stack(e_canon_year['EMBEDDING'].values)
        # get the cosine similarities
        historical_cosine_similarities.append(cosine_similarity(historical_embeddings_year).mean())
        e_canon_cosine_similarities.append(cosine_similarity(e_canon_embeddings_year).mean())
        # get the absolute difference
        years.append(year)

# plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, historical_cosine_similarities, label='Historical novels')
ax.plot(years, e_canon_cosine_similarities, label='Canon novels')
ax.set_xlabel("Year")
ax.set_ylabel("Mean cosine similarity")
ax.legend()


# %%

# make sure we dropped the row that does not have metadata
df = df.dropna(subset=['YEAR'])
df['YEAR'] = df['YEAR'].astype(int)

# get a df with just the rows in groups
hist_and_canon = df.loc[df['CATEGORY'].isin(['HISTORICAL', 'CANON_HISTORICAL', 'LEX_CANON', 'CE_CANON'])]

## Start a loop over the years
mean_similarity_dict = {}

# window size and step size
window_size = 5
step_size = 1

# set sampling
sampling = False
sample_size = 4

number_of_runs = 1

# Get the minimum and maximum years in the dataset
min_year = df['YEAR'].min()
max_year = df['YEAR'].max()

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
        subset = df.loc[(df['YEAR'].isin(year_range)) & (df['CATEGORY'] != 'O')] #hist_and_canon.loc[hist_and_canon['YEAR'].isin(year_range)]
        # HISTORICAL
        historical = df.loc[(df['YEAR'].isin(year_range)) & (df['CATEGORY'].isin(['HISTORICAL', 'CANON_HISTORICAL']))]
        # CANON
        e_canon = df.loc[(df['YEAR'].isin(year_range)) & (df['E_CANON'] == 1)]
        # OTHER
        # get whats in the range and is 'O'
        others = df.loc[(df['YEAR'].isin(year_range)) & (df['CATEGORY'] == 'O')]
        # NONCANON
        non_canon = df.loc[(df['YEAR'].isin(year_range)) & (df['E_CANON'] != 1)]
        # TOTAL
        df_total = df.loc[df['YEAR'].isin(year_range)]

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
            embeddings = np.stack(data['EMBEDDING'].values)
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
        historical_mean = group_eb['historical']['EMBEDDING'].mean(axis=0)
        e_canon_mean = group_eb['e_canon']['EMBEDDING'].mean(axis=0)
        others_mean = group_eb['other']['EMBEDDING'].mean(axis=0)
        non_canon_mean = group_eb['non_canon']['EMBEDDING'].mean(axis=0)

        # get the cosine similarity btw canon and non-canon
        canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, e_canon_mean])).mean()
        temp['CANON_NONCANON_COSIM'] = canon_noncanon_similarity

        # get the cosine similarity btw historicals and non-canon
        hist_noncanon_similarity = cosine_similarity(np.stack([historical_mean, others_mean])).mean()
        temp['HIST_OTHERS_COSIM'] = hist_noncanon_similarity

        hist_canon_similarity = cosine_similarity(np.stack([historical_mean, e_canon_mean])).mean()
        temp['HIST_CANON_COSIM_MEAN'] = hist_canon_similarity

        # save
        mean_similarity_dict[range_label] = temp
        #historical_cosine_similarity, e_canon_cosine_similarity, subset_cosine_similarity, others_similarity, non_canon_similarity, total_similarity, canon_noncanon_similarity, hist_noncanon_similarity, [len(df_total), len(others), len(subset), len(historical), len(e_canon)]

    # done

    # Format df
    sim_df = pd.DataFrame.from_dict(mean_similarity_dict, orient='index').reset_index()
    sim_df = sim_df.rename(columns={"index": "YEAR_RANGE"})
    # sim_df = sim_df.rename(columns={
    #     "index": "YEAR_RANGE", 0: "WITHIN_HIST_SIMILARITY", 1: "WITHIN_CANON_SIMILARITY", 2: "HIST_CANON_SIMILARITY", 3: "O_SIMILARITY", 4: "NONCANON_SIMILARITY", 5: "FULL_CORPUS_SIMILARITY", 
    #     6: "C_NC_SIMILARITY", 7:"HIST_NC_SIMILARITY", 8: "N_BOOKS (TOTAL, O, HIST_CANON, HIST, CANON)"})

    sim_df['START_YEAR'] = sim_df['YEAR_RANGE'].apply(lambda x: int(x.split('-')[0]))

    # append df to list
    runs_list.append(sim_df)
    sim_df.head(20)


# %%
# plot 
# make 3 plots on a row with the similarities
sim_cols = ['CANON_COSIM_MEAN', 'NONCANON_COSIM_MEAN', 'TOTAL_COSIM_MEAN'] # 'HIST_CANON_COSIM_MEAN', 'HIST_COSIM_MEAN', 'OTHERS_COSIM_MEAN', 
colors = ['#75BCC6', '#BA5C12', 'grey', 'purple'] # 'orange', 'yellow',
labels = ['E_CANON', 'HISTORICAL/CANON', 'NON_CANON', 'ALL']


correlation_vals = {col: [] for col in sim_cols}

fig, axs = plt.subplots(1, len(sim_cols), figsize=(23, 4), dpi=500)

for idx, frame in enumerate(runs_list):

    corrs = []

    for i, col in enumerate(sim_cols):
        axs[i].plot(frame['START_YEAR'], frame[col], label=labels[i], color=colors[i], alpha=0.5, linewidth=1)
        axs[i].set_xlabel("t")
        axs[i].set_ylabel(r"$\overline{x}$ cosine similarity")
        #axs[i].legend()

        # also get correlation and make title
        corr, pval = spearmanr(frame['START_YEAR'], frame[col])
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

# %%
# plot c/nc & h/nc similarity

plt.figure(figsize=(7, 4))

corrs = []

for i, frame in enumerate(runs_list):

    plt.plot(frame['START_YEAR'], frame['CANON_NONCANON_COSIM'], label='Canon vs non-canon similarity', color='#75BCC6', linewidth=2, 
             alpha=0.3)
    # get spearmanr
    corr, pval = spearmanr(frame['START_YEAR'], frame['CANON_NONCANON_COSIM'])
    print(f"CORR = {round(corr, 2)}** {round(pval,2)}")

    corrs.append(corr)

    # get the correlation after the year
    year = 1875
    sim_df_after_1874 = frame.loc[frame['START_YEAR'] >= year]
    corr_threshold, pval_threshold = spearmanr(sim_df_after_1874['START_YEAR'], sim_df_after_1874['CANON_NONCANON_COSIM'])
    print(f"after {year}: {round(corr_threshold,2)}, p-value: {round(pval_threshold,2)}")

plt.title(f"Spearman $r$ = {round(corr, 2)}**")
plt.show()

# %%



# we want to do the same as above, but instead of sampling, we want to simulate a sitribution for each window
# functions for simulation

# INTRAGROUP

def simulate_cosim_mean_std(key, num_simulations):
        data = group_eb[key]
        if len(data) > 1:
            embeddings = np.stack(data['EMBEDDING'].values)
            cosim_matrix = cosine_similarity(embeddings)
            # exclude the diagonal of the matrix
            cosim_values = cosim_matrix[np.triu_indices_from(cosim_matrix, k=1)]

            simulated_means = []
            for _ in range(num_simulations):
                # Simulate data from a normal distribution based on the sample mean and std
                simulated_data = np.random.normal(cosim_values.mean(), cosim_values.std(), len(cosim_values))
                simulated_means.append(simulated_data.mean())
            return np.mean(simulated_means), np.std(simulated_means)
        else:
            return np.nan, np.nan
    
        
# INTERGROUP

def simulate_cosim_between_groups(key1, key2, num_simulations):
    # Get embeddings for each group
    data1 = group_eb[key1]
    data2 = group_eb[key2]
    
    if len(data1) > 1 and len(data2) > 1:
        embeddings1 = np.stack(data1['EMBEDDING'].values)
        embeddings2 = np.stack(data2['EMBEDDING'].values)

        mean_emb1_list = []
        mean_emb2_list = []

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
    else:
        return np.nan    


# comment above and uncomment below functions to try w bootstrap resampling

# # INTRAGROUP

# def simulate_cosim_mean_std(key, num_simulations):
#     data = group_eb[key]
#     if len(data) > 1:
#         embeddings = np.stack(data['EMBEDDING'].values)
#         cosim_matrix = cosine_similarity(embeddings)
        
#         # again, exclude the diagonal
#         cosim_values = cosim_matrix[np.triu_indices_from(cosim_matrix, k=1)]
        
#         bootstrap_means = []
#         for _ in range(num_simulations):
#             # Generate a bootstrap sample with replacement
#             bootstrap_sample = resample(cosim_values)
#             # Calculate the mean of the bootstrap sample
#             bootstrap_mean = np.mean(bootstrap_sample)

#             bootstrap_means.append(bootstrap_mean)        
#         return np.mean(bootstrap_means), np.std(bootstrap_means)
#     else:
#         return np.nan, np.nan
  

# # INTERGROUP


# def simulate_cosim_between_groups(key1, key2, num_simulations):
#     # Get embeddings for each group
#     data1 = group_eb[key1]
#     data2 = group_eb[key2]
    
#     if len(data1) > 1 and len(data2) > 1:
#         embeddings1 = np.stack(data1['EMBEDDING'].values)
#         embeddings2 = np.stack(data2['EMBEDDING'].values)
        
#         mean_emb1_list = []
#         mean_emb2_list = []

#         for _ in range(num_simulations):
#             # Sample with replacement from both groups independently
#             bootstrap_sample1 = resample(embeddings1)
#             bootstrap_sample2 = resample(embeddings2)
            
#             # Calculate mean embeddings for each bootstrap sample
#             mean_embedding1 = np.mean(bootstrap_sample1, axis=0)
#             mean_embedding2 = np.mean(bootstrap_sample2, axis=0)

#             # mean embedding 1 and 2
#             mean_emb1_list.append(mean_embedding1)
#             mean_emb2_list.append(mean_embedding2)
        
#         # get overall mean of all simulation runs for each group
#         # so i get two means
#         overall_mean1 = np.mean(mean_emb1_list, axis=0)
#         overall_mean2 = np.mean(mean_emb2_list, axis=0)

#         # # Calculate cosine similarity between the mean embeddings
#         similarity = cosine_similarity([overall_mean1], [overall_mean2])


#         return similarity
#     else:
#         return np.nan
    


# SET PARAMETERS

# get a df with just the rows in groups
hist_and_canon = df.loc[df['CATEGORY'].isin(['HISTORICAL', 'CANON_HISTORICAL', 'LEX_CANON', 'CE_CANON'])]

mean_similarity_dict = {}

# window size and step size
window_size = 4
step_size = 1

# Number of simulations
num_simulations = 1000

# Get the minimum and maximum years in the dataset
min_year = df['YEAR'].min()
max_year = df['YEAR'].max()


# Loop over the range of years with a rolling window
for start_year in range(min_year, max_year - window_size + 1, step_size):

    temp = {}

    # Define rolling window range for each window
    year_range = list(range(start_year, start_year + window_size))
    range_label = f"{year_range[0]}-{year_range[-1]}"

    # Subset the data for the current window
    subset = df.loc[(df['YEAR'].isin(year_range)) & (df['CATEGORY'] != 'O')]
    historical = df.loc[(df['YEAR'].isin(year_range)) & (df['CATEGORY'].isin(['HISTORICAL', 'CANON_HISTORICAL']))]
    e_canon = df.loc[(df['YEAR'].isin(year_range)) & (df['E_CANON'] == 1)]
    others = df.loc[(df['YEAR'].isin(year_range)) & (df['CATEGORY'] == 'O')]
    non_canon = df.loc[(df['YEAR'].isin(year_range)) & (df['E_CANON'] != 1)]
    df_total = df.loc[df['YEAR'].isin(year_range)]

    # make subset dictionary per window for easier handling
    # empty it
    group_eb = {}
    group_eb = {'subset': subset, 'historical': historical, 'e_canon': e_canon,
                        'other': others, 'non_canon': non_canon, 'df_total': df_total}


    # HIST and CANON
    subset_mean, subset_std = simulate_cosim_mean_std('subset', num_simulations)
    temp['HIST_CANON_COSIM_MEAN'] = subset_mean
    temp['HIST_CANON_COSIM_STD'] = subset_std

    # HIST
    hist_mean, hist_std = simulate_cosim_mean_std('historical', num_simulations)
    temp['HIST_COSIM_MEAN'] = hist_mean
    temp['HIST_COSIM_STD'] = hist_std

    # CANON
    c_mean, c_std = simulate_cosim_mean_std('e_canon', num_simulations)
    temp['CANON_COSIM_MEAN'] = c_mean
    temp['CANON_COSIM_STD'] = c_std

    # OTHER
    o_mean, o_std = simulate_cosim_mean_std('other', num_simulations)
    temp['OTHERS_COSIM_MEAN'] = o_mean
    temp['OTHERS_COSIM_STD'] = o_std

    # NON-CANON
    nc_mean, nc_std = simulate_cosim_mean_std('non_canon', num_simulations)
    temp['NONCANON_COSIM_MEAN'] = nc_mean
    temp['NONCANON_COSIM_STD'] = nc_std

    # TOTAL
    t_mean, t_std = simulate_cosim_mean_std('df_total', num_simulations)
    temp['TOTAL_COSIM_MEAN'] = t_mean
    temp['TOTAL_COSIM_STD'] = t_std

    temp['N_BOOKS'] = [len(df_total), len(others), len(subset), len(historical), len(e_canon)]

    # Intergroup similarity calculations

    # first we get the mean per group, then we compare the means to each other
    # i.e., we want to treat each group differently, and then compare them

    # e_canon_mean = group_eb['e_canon']['EMBEDDING'].mean(axis=0)
    # non_canon_mean = group_eb['non_canon']['EMBEDDING'].mean(axis=0)
    #canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, e_canon_mean])).mean()

    mean_similarity_c_nc = simulate_cosim_between_groups('e_canon', 'non_canon', num_simulations)
    temp['CANON_NONCANON_COSIM'] = mean_similarity_c_nc

    mean_similarity_c_hist = simulate_cosim_between_groups('e_canon', 'historical', num_simulations)
    temp['HIST_CANON_COSIM'] = mean_similarity_c_hist


    # Save the results
    mean_similarity_dict[range_label] = temp

# Format the final dataframe
simulate_df = pd.DataFrame.from_dict(mean_similarity_dict, orient='index').reset_index()
simulate_df = simulate_df.rename(columns={"index": "YEAR_RANGE"})

simulate_df['START_YEAR'] = simulate_df['YEAR_RANGE'].apply(lambda x: int(x.split('-')[0]))

simulate_df.head(20)


print(num_simulations)
# %%

# plot 
# make 3 plots on a row with the similarities
sim_cols = ['CANON_COSIM_MEAN','TOTAL_COSIM_MEAN'] # 'HIST_CANON_COSIM_MEAN', 'HIST_COSIM_MEAN', 'OTHERS_COSIM_MEAN', 'NONCANON_COSIM_MEAN', 
colors = ['#75BCC6', 'grey', 'purple'] # 'orange', 'yellow','#BA5C12',
labels = ['E_CANON',  'ALL'] # 'HISTORICAL/CANON', 'NON_CANON',


correlation_vals = {col: [] for col in sim_cols}

fig, axs = plt.subplots(1, len(sim_cols), figsize=(18, 3), dpi=500)

corrs = []

for i, col in enumerate(sim_cols):
    axs[i].plot(simulate_df['START_YEAR'], simulate_df[col], label=labels[i], color=colors[i], alpha=0.5, linewidth=3)
    axs[i].set_xlabel("t")
    axs[i].set_ylabel(r"$\overline{x}$ cosine similarity")
    #axs[i].legend()

    # also get correlation and make title
    corr, pval = spearmanr(simulate_df['START_YEAR'], simulate_df[col])
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

    plt.tight_layout()
        
    #correlation_vals[f'run_{idx}'] = dict(zip(sim_cols, corrs))

for i, col in enumerate(sim_cols):
    avg_corr = np.mean(correlation_vals[col])  # Calculate the average correlation
    axs[i].set_title(f"{col.split('_')[0]}, Average Spearman $r$ = {round(avg_corr, 2)}", fontsize=12)


plt.show()

# %%
# plot c/nc & h/nc similarity

plt.figure(figsize=(8.5, 3), dpi=500)
plt.plot(simulate_df['START_YEAR'], simulate_df['CANON_NONCANON_COSIM'], label='Canon vs non-canon similarity', color='#75BCC6', linewidth=4, 
            alpha=0.5)
# get spearmanr
corr, pval = spearmanr(simulate_df['START_YEAR'], simulate_df['CANON_NONCANON_COSIM'])
print(f"CORR = {round(corr, 2)}** {round(pval,2)}")

# get the correlation after the year XX
year = 1875
sim_df_after_1874 = simulate_df.loc[simulate_df['START_YEAR'] >= year]
corr_threshold, pval_threshold = spearmanr(sim_df_after_1874['START_YEAR'], sim_df_after_1874['CANON_NONCANON_COSIM'])
print(f"after {year}: {round(corr_threshold,2)}, p-value: {round(pval_threshold,2)}")

plt.tight_layout()
plt.title(f"Canon/Non_canon, Spearman $r$ = {round(corr, 2)}**")
plt.show()

# %%
# same for hist/canon
plt.figure(figsize=(7, 4))
plt.plot(simulate_df['START_YEAR'], simulate_df['HIST_CANON_COSIM'], label='Historical vs canon similarity', color='#FE7F2D', linewidth=4, 
            alpha=0.5)
# get spearmanr
corr, pval = spearmanr(simulate_df['START_YEAR'], simulate_df['HIST_CANON_COSIM'])
print(f"CORR = {round(corr, 2)}** {round(pval,2)}")
plt.title(f"Historical/Canon, Spearman $r$ = {round(corr, 2)}**")
plt.show()


# %%
len(simulate_df)

# %%

# PART IV: we want to know the direction of similarity

sorted_df_year = df.sort_values(by='YEAR', ascending=True)
print(len(df))

threshold = 419

# gte the first 419 in a df and the last in another
df_first_15 = sorted_df_year.iloc[:threshold]
df_last_15 = sorted_df_year.iloc[threshold:]

# get the mean embedding of the canon group pre/past threshold
e_canon_first = df_first_15.loc[df_first_15['E_CANON'] == 1]
e_canon_last = df_last_15.loc[df_last_15['E_CANON'] == 1]
non_canon_first = df_first_15.loc[df_first_15['E_CANON'] != 1]
non_canon_last = df_last_15.loc[df_last_15['E_CANON'] != 1]

# lets plot into the same axes

e_canon_first_array = np.stack(e_canon_first['EMBEDDING'].values).mean(axis=0)
e_canon_last_array = np.stack(e_canon_last['EMBEDDING'].values).mean(axis=0)
non_canon_first_array = np.stack(non_canon_first['EMBEDDING'].values).mean(axis=0)
non_canon_last_array = np.stack(non_canon_last['EMBEDDING'].values).mean(axis=0)

# Stack the mean embeddings into one array
mean_embeddings = [e_canon_first_array, e_canon_last_array, non_canon_first_array, non_canon_last_array]
labels = ['Canon First', 'Canon Last', 'Non-Canon First', 'Non-Canon Last']

pca = PCA(n_components=2)
mean_embeddings_2d = pca.fit_transform(mean_embeddings)

# Plot the embeddings
plt.figure(figsize=(8, 8))
plt.scatter(mean_embeddings_2d[:, 0], mean_embeddings_2d[:, 1], color=['#75BCC6', '#75BCC6', '#BA5C12', '#BA5C12'], s=1000, alpha=0.6)
for i, txt in enumerate(labels):
    plt.annotate(txt, (mean_embeddings_2d[i, 0], mean_embeddings_2d[i, 1]))
plt.show()




# %%

dat = df[['EMBEDDING', 'CATEGORY', 'YEAR', 'E_CANON']].copy()

sorted_df_year = dat.sort_values(by='YEAR', ascending=True).reset_index(drop=True)

for i, row in sorted_df_year.iterrows():
    # make column
    if i < threshold and row['E_CANON'] == 1:
        sorted_df_year.loc[i, 'CAT_COMPLEX'] = 'C_FIRST'
    elif i < threshold and row['E_CANON'] != 1:
        sorted_df_year.loc[i, 'CAT_COMPLEX'] = 'NC_FIRST'
    else:
        if i>= threshold and row['E_CANON'] == 1:
            sorted_df_year.loc[i, 'CAT_COMPLEX'] = 'C_LAST'
        elif i >= threshold and row['E_CANON'] != 1:
            sorted_df_year.loc[i, 'CAT_COMPLEX'] = 'NC_LAST'

sorted_df_year.head()


#color_mapping = {'NC_LAST': '#7F3E0C', 'NC_FIRST': '#F7B267', 'C_FIRST': '#75BCC6', 'C_LAST': '#2A4C5E'}
color_mapping = {'NC_LAST': '#129525', 'NC_FIRST': '#A1E44D', 'C_FIRST': '#7AFDD6', 'C_LAST': '#2A4C5E'}

color_mapping_extend = {'NC_LAST_mean': '#129525', 'NC_FIRST_mean': '#A1E44D', 'C_FIRST_mean': '#7AFDD6', 'C_LAST_mean': '#2A4C5E'}



def plot_pca_c_nc(data, title, colormapping):

    # Handle embeddings
    new_rows = []

    for category in data["CAT_COMPLEX"].unique():
        mean_embeddings = np.stack(data[data["CAT_COMPLEX"] == category]["EMBEDDING"].to_list()).mean(axis=0)
        #data = data.append({"EMBEDDING": mean_embeddings, "CAT_COMPLEX": category + 'mean'}, ignore_index=True)
        new_row = {"EMBEDDING": mean_embeddings, "CAT_COMPLEX": category + '_mean'}
        new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    data = pd.concat([data, new_rows_df], ignore_index=True)


    embeddings_array = np.array(data["EMBEDDING"].to_list(), dtype=np.float32)
    
    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["CAT_COMPLEX"] = data["CAT_COMPLEX"].values

    # We're gonna set a different alpha for the 'LAST' groups
    alpha_dict = dict(zip(color_mapping.keys(), [0.45 if x.endswith('LAST') else 0.3 for x in color_mapping.keys()]))
    #markers_dict = dict(zip(colormapping.keys(), ['x' if x.endswith('mean') else 'o' for x in color_mapping.keys()]))


    # Plot each category
    for category in df_pca["CAT_COMPLEX"].unique():
        if category.endswith('mean') is False:
            subset = df_pca[df_pca["CAT_COMPLEX"] == category]


            alpha = alpha_dict.get(category)
            #marker = markers_dict.get(category)
            
            plt.scatter(
                subset["PCA1"],
                subset["PCA2"],
                color=color_mapping.get(category),
                label=category,
                alpha=alpha,
                edgecolor='black',
                s=100,
                marker='o'
            )

    # plot mean points
    for category in df_pca["CAT_COMPLEX"].unique():
        if category.endswith('mean') is True:
            subset = df_pca[df_pca["CAT_COMPLEX"] == category]

            alpha = alpha_dict.get(category)
            #marker = markers_dict.get(category)

            plt.scatter(
                subset["PCA1"],
                subset["PCA2"],
                color=color_mapping_extend.get(category),
                label=category,
                alpha=0.9,
                edgecolor='black',
                s=300,
                marker='o'
            )
    legend_handles = [Patch(facecolor=color, label=label) for label, color in color_mapping.items()]
    plt.legend(handles=legend_handles, loc='upper right')


plt.figure(figsize=(8, 8))
plot_pca_c_nc(sorted_df_year, "First 15 years", color_mapping)
# reduce fontsize label
#plt.legend(fontsize=9, loc='upper right')


plt.show()


# %%
# same but trying to plot the outlines of clusters
# might be more readable if we plot the convex hulls of the clusters

from scipy.spatial import ConvexHull

def plot_pca_c_nc(data, title, colormapping):

    # Handle embeddings
    new_rows = []
    for category in data["CAT_COMPLEX"].unique():
        mean_embeddings = np.stack(data[data["CAT_COMPLEX"] == category]["EMBEDDING"].to_list()).mean(axis=0)
        new_row = {"EMBEDDING": mean_embeddings, "CAT_COMPLEX": category + '_mean'}
        new_rows.append(new_row)
    new_rows_df = pd.DataFrame(new_rows)
    data = pd.concat([data, new_rows_df], ignore_index=True)

    embeddings_array = np.array(data["EMBEDDING"].to_list(), dtype=np.float32)
    
    # PCA to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    df_pca["CAT_COMPLEX"] = data["CAT_COMPLEX"].values

    # Set alpha for the 'LAST' groups
    alpha_dict = dict(zip(color_mapping.keys(), [0.45 if x.endswith('LAST') else 0.3 for x in color_mapping.keys()]))

    # Plot each category
    for category in df_pca["CAT_COMPLEX"].unique():
        subset = df_pca[df_pca["CAT_COMPLEX"] == category]
        alpha = alpha_dict.get(category)
        plt.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=color_mapping.get(category),
            label=category,
            alpha=alpha,
            edgecolor='black',
            s=100,
            marker='o'
        )
        
    
        # Compute and plot convex hull
        if len(subset) > 2:
            points = subset[["PCA1", "PCA2"]].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], color=color_mapping.get(category), alpha=0.5, linewidth=2)

    # Plot the mean points
    for category in df_pca["CAT_COMPLEX"].unique():
        if category.endswith('mean'):
            subset = df_pca[df_pca["CAT_COMPLEX"] == category]
            plt.scatter(
                subset["PCA1"],
                subset["PCA2"],
                color=color_mapping_extend.get(category),
                label=category,
                alpha=0.9,
                edgecolor='black',
                s=300,
                marker='o'
            )

    legend_handles = [Patch(facecolor=color, label=label) for label, color in color_mapping.items()]
    plt.legend(handles=legend_handles, loc='upper right')
    plt.title(title)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')

plt.figure(figsize=(8, 8))
plot_pca_c_nc(sorted_df_year, "", color_mapping)
plt.show()
# %%



# We want to doi the exact same thing as above, but we want to plot a rolling window 
# we want to downsample majority group for each window, and get the mean embedding for each group

simple = df[['EMBEDDING', 'YEAR', 'E_CANON']].copy()
print(len(simple))

rolling_dict = {}

# window size and step size
window_size = 10
step_size = 3

# Get the minimum and maximum years in the dataset
min_year = df['YEAR'].min()
max_year = df['YEAR'].max()


# Loop over the range of years with a rolling window
for start_year in range(min_year, max_year - window_size + 1, step_size):

    temp = {}

    # Define rolling window range for each window
    year_range = list(range(start_year, start_year + window_size))
    range_label = f"{year_range[0]}-{year_range[-1]}"

    # Subset the data for the current window
    e_canon = simple.loc[(simple['YEAR'].isin(year_range)) & (simple['E_CANON'] == 1)]
    non_canon = simple.loc[(simple['YEAR'].isin(year_range)) & (simple['E_CANON'] != 1)]

    # downsample the majority group
    if len(e_canon) < len(non_canon):
        non_canon_sample = non_canon.sample(len(e_canon), random_state=42)
    else:
        non_canon_sample = non_canon


    window_mean_canon = np.stack(e_canon['EMBEDDING'].values).mean(axis=0)
    window_mean_non_canon = np.stack(non_canon_sample['EMBEDDING'].values).mean(axis=0)
    
    rolling_dict[range_label] = {'e_canon_emb': e_canon['EMBEDDING'].values, 'non_canon_emb': non_canon_sample['EMBEDDING'].values, 'e_canon_mean': window_mean_canon, 'non_canon_mean': window_mean_non_canon, 'datalengths': [len(e_canon), len(non_canon_sample)]}

# check
print('no of windows:', len(rolling_dict))
pd.DataFrame.from_dict(rolling_dict, orient='index').head(20)


# %%

# Plot

# Generate distinguishable colors with wider spacing
num_windows = len(rolling_dict.keys())
blues = sns.color_palette("Blues", n_colors=num_windows, desat=1)
greens = sns.color_palette("Reds", n_colors=num_windows, desat=1)

# Color mapping for individual points
color_mapping_e_canon = {window: blues[i] for i, window in enumerate(rolling_dict.keys())}
color_mapping_non_canon = {window: greens[i] for i, window in enumerate(rolling_dict.keys())}

# Color mapping for mean points (extend the individual mappings)
color_mapping_e_canon_mean = {window: blues[i] for i, window in enumerate(rolling_dict.keys())}
color_mapping_non_canon_mean = {window: greens[i] for i, window in enumerate(rolling_dict.keys())}

# Fit PCA on all embeddings
all_embeddings = np.stack(simple['EMBEDDING'].values)
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
plt.figure(figsize=(12, 8))

# # Scatter plot for individual embeddings and convex hulls
# for window in pca_df['window'].unique():
#     for group in ['e_canon', 'non_canon']:
#         subset = pca_df[(pca_df['window'] == window) & (pca_df['group'] == group)]
#         plt.scatter(
#             subset['PCA1'], subset['PCA2'], 
#             color=subset['color'].values[0], alpha=0.3, edgecolor='black', s=100
#         )
        
#         if len(subset) > 2:  # Convex hull requires at least 3 points
#             points = subset[['PCA1', 'PCA2']].values
#             hull = ConvexHull(points)
#             for simplex in hull.simplices:
#                 plt.plot(points[simplex, 0], points[simplex, 1], color=subset['color'].values[0], alpha=0.3, linewidth=3)

# Scatter plot for mean points
for group in ['e_canon_mean', 'non_canon_mean']:
    subset = pca_df[pca_df['group'] == group]
    plt.scatter(
        subset['PCA1'], subset['PCA2'],
        color=subset['color'].values, alpha=0.45, edgecolor='black', s=500, marker='o'
    )

    # annotate the mean points with numbers corresponding to their order
    numbers = list(range(1, len(subset) + 1))
    numbers_formatted = ['('+x+')' for x in map(str, numbers)]
    for i, txt in enumerate(numbers_formatted):
        plt.annotate(txt, (subset['PCA1'].values[i], subset['PCA2'].values[i]), fontsize=9, color='black', ha='center') #weight='bold',

    # for i, txt in enumerate(subset['window']):
    #     plt.annotate(txt, (subset['PCA1'].values[i], subset['PCA2'].values[i]))

    # get labels
    label = {'Canon': 'blue', 'Non-Canon': 'green'}
    legends = [Patch(facecolor=color, label=label) for label, color in color_mapping_e_canon.items()]


plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
# %%
legends
# %%
