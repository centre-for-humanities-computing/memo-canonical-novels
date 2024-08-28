# prompt is 87 tokens, max-tokens = 512-87
python src/embeddings.py \
    --model-name "intfloat/multilingual-e5-large-instruct" \
    --max-tokens 425 \
    --prefix "Instruct: Identify the author of a given passage from historical Danish fiction\nQuery: " \
    --prefix-description "identify_author"

# prompt is 89 tokens, max-tokens = 512-89
python src/embeddings.py \
    --model-name "intfloat/multilingual-e5-large-instruct" \
    --max-tokens 423 \
    --prefix "Instruct: Given a passage from historical Danish fiction, retrieve similar texts \nQuery: " \
    --prefix-description "retrieve"