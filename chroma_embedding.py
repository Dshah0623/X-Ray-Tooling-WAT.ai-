import json
from cohere import Client
import os
import pandas as pd
import dotenv
import csv
import argparse
import langchain
from langchain.embeddings import HuggingFaceEmbeddings
import time
import numpy as np
import pickle
from scipy import spatial
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader


class ChromaEmbedding():
    __open_key = os.getenv('OPENAI_API_KEY')
    __embeddings_hugging = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    __embedding_open = OpenAIEmbeddings(openai_api_key=__open_key)
    __persist_chroma_directory = 'db'

    def __init__(self) -> None:
        self.__xray_articles = self.__load_xray_articles()
        self.__xray_chunked_articles = self.__chunk_documents(
            self.__xray_articles)
        self.__embedding_in_use = None
        self.__chroma_db = None

    def set_embedding_model(self, use_hugging_face=False, use_open_ai=False) -> None:
        if use_hugging_face:
            self.embedding_in_use = self.__embeddings_hugging
        elif use_open_ai:
            self.embedding_in_use = self.__embedding_open
        else:
            raise ValueError("No embedding model selected")

    def __load_and_chunk_articles(self) -> object:
        docs = self.__load_xray_articles()
        self.__xray_chunked_articles = self.__chunk_documents(docs)

    def __load_xray_articles(self) -> object:
        loader = JSONLoader(
            file_path="datasets/xray_articles_processed.json",
            jq_schema='.[].FullText',
            text_content=True)

        return loader.load()

    def __chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200) -> object:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    def create_and_populate_chroma(self) -> None:

        vector_db = Chroma.from_documents(documents=self.__xray_chunked_articles,
                                          embedding=self.__embedding_in_use,
                                          persist_directory=self.__persist_chroma_directory)

        vector_db.persist()

        # self.chroma_db = vector_db

    def __create_ids(self) -> list:
        ids = [str(i) for i in range(1, len(self.__xray_chunked_articles) + 1)]

    def load_chroma_db(self) -> None:
        vector_db = Chroma(persist_directory=self.__persist_chroma_directory,
                           embedding_function=self.embedding_in_use, ids=self.__create_ids())

        self.__chroma_db = vector_db

    def retrieve_from_chroma(self, query, search_kwargs=2) -> object:
        # retriever = self.chroma_db.as_retriever()
        # retriever = self.__chroma_db.as_retriever(
        #     search_kwargs={"k": search_kwargs})
        # docs = retriever.get_relevant_documents(query)

        docs = self.__chroma_db.similarity_search(query)
        return docs

    def reupload_to_chroma(self) -> None:
        self.clear_chroma()
        self.__load_and_chunk_articles()
        self.__chroma_db = Chroma.from_documents(
            self.__xray_chunked_articles, ids=self.__create_ids())

    def clear_chroma(self) -> None:
        ids_to_delete = []

        for doc in self.__chroma_db:
            if doc.metadata.get('source') == self.__xray_chunked_articles:
                ids_to_delete.append(doc.id)

        self.__chroma_db.delete(ids=ids_to_delete)

    def get_articles(self) -> object:
        return self.__xray_articles

    def get_chunked_articles(self) -> object:
        return self.__xray_chunked_articles

    def get_hugging_embedding_model(self) -> object:
        return self.__embeddings_hugging

    def get_open_embedding_model(self) -> object:
        return self.__embedding_open

    def get_current_embedding_model(self) -> object:
        return self.__embedding_in_use

    def get_chroma_db(self) -> object:
        return self.__chroma_db

# if __name__ == 'main':
#     c = ChromaEmbedding()
#     argparse...
#     if flag == '--build':
#         embeddings = c.create_embeddings()
#         c.populate_embeddings(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser
