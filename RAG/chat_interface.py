from cohere import Client
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import uuid
import argparse
import pandas as pd
from chroma_embedding import ChromaEmbedding
from index_embedding import IndexEmbedding

class Chat(ABC):
    @abstractmethod
    def query(self, query):
        """
        run query through the chat to get a response with rag
        """
        pass

class Cohere(Chat):
    __cohere_key = os.getenv('COHERE_API_KEY')
    __co = Client(__cohere_key)
    __conversation_id = str(uuid.uuid4())

    def __init__(self, chroma_embedding=True, use_openai=False, chunking_max_tokens=100, num_matches=5, dataset_path="RAG/datasets/"):
        """
        max_tokens is the max number of words in a response from the api
        """
        self.__embedding = None
        if chroma_embedding:
            self.__embedding = ChromaEmbedding(
                 use_openai=use_openai,
                 num_matches=num_matches,
                 dataset_path=dataset_path
            )
        else:
            self.__embedding = IndexEmbedding(
                 use_openai=use_openai,
                 chunking_max_tokens=chunking_max_tokens,
                 num_matches=num_matches,
                 dataset_path=dataset_path
            )

    def query(self, query):
        rag_docs = self.__embedding.get_similar_documents(query, self.__max_docs)

        response = self.__co.chat(
            message=query,
            documents=rag_docs,
            conversation_id=self.__conversation_id,
            max_tokens=self.__max_tokens,
        )
        return response.text
    
    def end_chat(self):
        self.__embedding.destroy()
    
class OpenAI(Chat):
    __open_api_key = os.getenv('OPENAI_API_KEY')
    __open_llm = OpenAI(temperature=0, openai_api_key=__open_api_key)

    def __init__(self, chroma_embedding=True, use_openai=False, chunking_max_tokens=100, num_matches=5, dataset_path="RAG/datasets/"):
        """
        max_tokens is the max number of words in a response from the api
        """
        self.__embedding = None
        if chroma_embedding:
            self.__embedding = ChromaEmbedding(
                 use_openai=use_openai,
                 num_matches=num_matches,
                 dataset_path=dataset_path
            )
        else:
            self.__embedding = IndexEmbedding(
                 use_openai=use_openai,
                 chunking_max_tokens=chunking_max_tokens,
                 num_matches=num_matches,
                 dataset_path=dataset_path
            )

    def query(self, query):
        rag_docs = self.__embedding.get_similar_documents(query)

        chain = load_qa_chain(self.__open_llm, chain_type="stuff")
        out = chain.run(input_documents=rag_docs, question=query)
        return out
    
    def end_chat(self):
        self.__embedding.destroy()

if __name__=="__main__":
    # TODO - implement the argparse and actual driver to run a chat
    pass
