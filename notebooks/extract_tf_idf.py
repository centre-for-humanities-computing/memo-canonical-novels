# %%
from utils import *

# %%
# get the filenames and the texts to a df
path = '$PATH_TO_NORM'
file_dict = {}

# loop through the folder
for filename in os.listdir(path):
    print(filename)
    with open(path + '/' + filename, 'r') as file:
        text = file.read()
        print(text[:4])
        file_dict[filename] = text

print('length of file_dict:', len(file_dict))
# %%
# test file
file_dict['1881_Kielland_Else.txt']
# %%
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
    # and remove all digits
    file_dict[filename] = re.sub(r'\d', '', file_dict[filename])

print(file_dict['1881_Kielland_Else.txt'])

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
cols_to_keep = list(tfidf_df.columns)[:-170]
tfidf_df = tfidf_df[cols_to_keep].copy()
print(tfidf_df.shape)

# check that the last few columns make sense
list(tfidf_df.columns)[-10:]

# we want to know how many nan values there are (and why)
nans_per_column = tfidf_df.isna().sum()
print(nans_per_column)

# %%
# make the filename column normalised so we can merge
tfidf_df['FILENAME'] = tfidf_df.index.str.replace('.txt', '')
tfidf_df = tfidf_df.reset_index(drop=True)
tfidf_df.head()
# %%
# get embeddings df to merge with
path = 'data/emb_dataset'

ds = Dataset.load_from_disk(path)
df_emb = ds.to_pandas()
df_emb.columns = ['FILENAME', 'EMBEDDING', 'N_CHUNKS_ORIG']
print(len(df_emb))
df_emb.head()

# %%
# merge it
dt = df_emb[['FILENAME', 'EMBEDDING']]
dat = dt.merge(tfidf_df, on='FILENAME', how='left')
dat.head()
# %%
# we need to make the tf-idf values into a list (so theyre a vector in one column)
tfidf_columns = list(dat.columns)[2:] # choose all cols (tfidf cols) except the first two (filename and embedding)

dat['TF_IDF']= dat[tfidf_columns].values.tolist()
dat.head()

# %%
# # dump to dataset
# ds = Dataset.from_pandas(dat)
# ds.save_to_disk('data/tfidf_dataset')

dat = dat[['FILENAME', 'EMBEDDING', 'TF_IDF']].copy()
dat.head()

# %%
# # dump it to json
# with open('data/tfidf_dataset.json', 'w') as f:
#     json.dump(dat.to_dict(orient='records'), f)

# dump to json
dat.to_json('data/tfidf_dataset.json', orient='records')
# %%
print('All done!')
