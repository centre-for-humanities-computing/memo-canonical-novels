python src/embeddings.py \
    --model-name "MiMe-MeMo/MeMo-BERT-03" \
    --max-tokens 512

python src/embeddings.py \
    --model-name "KennethEnevoldsen/dfm-sentence-encoder-large-exp2-no-lang-align" \
    --max-tokens 512

python src/embeddings.py \
    --model-name "intfloat/multilingual-e5-large" \
    --max-tokens 510 \
    --prefix "Query: " \
    --prefix-description ""

# prompt is 87 tokens, max-tokens = 512-87
python src/embeddings.py \
    --model-name "intfloat/multilingual-e5-large-instruct" \
    --max-tokens 425 \
    --prefix "Instruct: Identify the author of a given passage from historical Danish fiction\nQuery: " \
    --prefix-description "identify_author"

# prompt is 91 tokens, max-tokens = 512-89
python src/embeddings.py \
    --model-name "intfloat/multilingual-e5-large-instruct" \
    --max-tokens 421 \
    --prefix "Instruct: Retrieve semantically similar texts given a passage in historical Danish \nQuery: " \
    --prefix-description "retrieve"
