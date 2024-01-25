if ! [ -f ./sqlite] then 
    python RAG/chroma_embedding.py --set_embedding_model --openai
    python RAG/chroma_embedding.py --build_chroma