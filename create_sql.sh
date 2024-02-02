#!/bin/bash

# Check if the sqlite file exists
if [ ! -f "./sqlite" ]; then
    # If the file does not exist, execute the Python commands
    python RAG/chroma_embedding.py set_model --openai
    python RAG/chroma_embedding.py build_chroma
fi