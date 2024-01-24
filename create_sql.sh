if ! [ -f ./sqlite] then 
    python chroma_embedding.py --set_embedding_model --openai
    python chroma_embedding.py --build_chroma