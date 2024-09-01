# %%
from utils import *
# %%
# get data
path = 'emb_multilingual_raw'

from datasets import Dataset

ds = Dataset.load_from_disk(path)
df_emb = ds.to_pandas()

print(len(df_emb))
df_emb.head()

# %%
list_emb = []

for i, row in df_emb[:1].iterrows():
    print(row['embedding'])
    for emb in row['embedding']:
        list_emb.append(emb)

print(len(list_emb))
# %%
len(list_emb[1])
# %%

# take the 20th one of 10 novels

emb_chunk_no_20 = {}
for i, row in df_emb[:10].iterrows():
    emb_chunk_no_20[row['novel']] = row['embedding'][20]

emb_chunk_no_20

# %%
# check they are all the same length
for key in emb_chunk_no_20.keys():
    print(len(emb_chunk_no_20[key]))
# %%
# we want to do cosine similarity between the 20th chunk of each novel

emb_matrix = np.stack(list(emb_chunk_no_20.values()))
cosine_matrix = cosine_similarity(emb_matrix)

cosine_matrix
# %%
# make heatmap
sns.heatmap(cosine_matrix, annot=True, xticklabels=emb_chunk_no_20.keys(), yticklabels=emb_chunk_no_20.keys())
# %%




# %%
# EMBEDDING DFM
path = 'dfm_raw'

from datasets import Dataset

ds = Dataset.load_from_disk(path)
df_emb = ds.to_pandas()

print(len(df_emb))
df_emb.head()

# %%
# take the 20th one of 10 novels

emb_chunk_no_20 = {}
for i, row in df_emb[:10].iterrows():
    emb_chunk_no_20[row['novel']] = row['embedding'][20]

emb_chunk_no_20

# %%
# check they are all the same length
for key in emb_chunk_no_20.keys():
    print(len(emb_chunk_no_20[key]))
# %%
# we want to do cosine similarity between the 20th chunk of each novel

emb_matrix = np.stack(list(emb_chunk_no_20.values()))
cosine_matrix = cosine_similarity(emb_matrix)

cosine_matrix
# %%
# make heatmap
sns.heatmap(cosine_matrix, annot=True, xticklabels=emb_chunk_no_20.keys(), yticklabels=emb_chunk_no_20.keys(), cbar=False)
# %%