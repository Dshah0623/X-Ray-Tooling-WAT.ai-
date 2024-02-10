from cohere import Client
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import uuid
import argparse
from chroma_embedding import ChromaEmbedding
from index_embedding import IndexEmbedding
from abc import ABC, abstractmethod

class Chat(ABC):
    @abstractmethod
    def query(self, query) -> object:
        """
        run query through the chat to get a response with rag
        """
        pass

    @abstractmethod
    def end_chat(self) -> None:
        """
        Cleans up resources
        """
        pass

class Cohere(Chat):
    """
    A class that integrates with the Cohere API for conversational AI purposes.

    Attributes:
        __cohere_key (str): API key for Cohere.
        __co (Client): Cohere client initialized with the API key.
        __conversation_id (str): Unique ID for tracking the conversation.
        __embedding Union(IndexEmbedding, ChromaEmbedding): an embedding system for the RAG to use
        __max_tokens int: max number of tokns a response to a query can be
    """
    __cohere_key = os.getenv('COHERE_API_KEY')
    __co = Client(__cohere_key)
    __conversation_id = str(uuid.uuid4())

    def __init__(self, chroma_embedding=True, use_openai=False, chunking_max_tokens=100, num_matches=5, max_tokens=500, dataset_path="RAG/datasets/"):
        """
        Initializes the Cohere class, setting up the embedding model used for queries.

        Args:
            chroma_embedding (bool): Flag to decide between using ChromaEmbedding or IndexEmbedding.
            use_openai (bool): Determines if OpenAI embeddings should be used vs huggingface.
            chunking_max_tokens (int): Maximum number of tokens for chunking in IndexEmbedding.
            num_matches (int): Number of matching documents to return upon a query.
            dataset_path (str [Path]): Path to the dataset directory.
        """
        self.__max_tokens = max_tokens
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

    def query(self, query) -> str:
        """
        Processes a query.

        Args:
            query (str): The query string.

        Returns:
            str: The text response from the Cohere API.
        """
        rag_docs = self.__embedding.get_similar_documents(query, self.__max_docs)

        response = self.__co.chat(
            message=query,
            documents=rag_docs,
            conversation_id=self.__conversation_id,
            max_tokens=self.__max_tokens,
        )
        return response.text
    
    def end_chat(self) -> None:
        """
        Cleans up resources
        """
        self.__embedding.destroy()
    
class OpenAI(Chat):
    """
    A class that integrates with OpenAI's language models for question answering purposes.

    Attributes:
        __open_api_key (str): API key for OpenAI.
        __open_llm (OpenAI): OpenAI language model initialized with the API key.
        __embedding Union(IndexEmbedding, ChromaEmbedding): an embedding system for the RAG to use
    """
    __open_api_key = os.getenv('OPENAI_API_KEY')
    __open_llm = OpenAI(temperature=0, openai_api_key=__open_api_key)

    def __init__(self, chroma_embedding=True, use_openai=False, chunking_max_tokens=100, num_matches=5, dataset_path="RAG/datasets/") -> None:
        """
        Initializes the OpenAI class, setting up the embedding model used for queries.

        Args:
            chroma_embedding (bool): Flag to decide between using ChromaEmbedding or IndexEmbedding.
            use_openai (bool): Determines if OpenAI embeddings should be used.
            chunking_max_tokens (int): Maximum number of tokens for chunking in IndexEmbedding.
            num_matches (int): Number of matching documents to return upon a query.
            dataset_path (str): Path to the dataset directory.
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

    def query(self, query) -> object:
        """
        Processes a query using OpenAI's language model.

        Args:
            query (str): The query string.

        Returns:
            str: The response from the language model.
        """
        rag_docs = self.__embedding.get_similar_documents(query)

        chain = load_qa_chain(self.__open_llm, chain_type="stuff")
        out = chain.run(input_documents=rag_docs, question=query)
        return out
    
    def end_chat(self) -> None:
        """
        Cleans up resources.
        """
        self.__embedding.destroy()

if __name__=="__main__":
    # TODO - implement the argparse and actual driver to run a chat
    pass
