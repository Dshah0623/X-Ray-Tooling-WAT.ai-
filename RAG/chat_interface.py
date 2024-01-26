from cohere import Client
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import uuid
import pandas as pd
from chroma_embedding import ChromaEmbedding

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

    def __init__(self, max_tokens=500, embedding_type="chroma", max_docs=5):
        """
        max_tokens is the max number of words in a response from the api
        """
        self.__max_tokens = max_tokens
        self.__max_docs = max_docs
        if embedding_type == "chroma":
            self.__embedding = ChromaEmbedding()
        else:
            self.__embedding = None # TODO fill in after for the vector index

    def query(self, query):
        rag_docs = self.__embedding.get_similar_documents(query, self.__max_docs)

        response = self.co.chat(
            message=query,
            documents=rag_docs,
            conversation_id=self.__conversation_id,
            max_tokens=self.__max_tokens,
        )
        return response.text
    
class OpenAI(Chat):
    __open_api_key = os.getenv('OPENAI_API_KEY')
    __open_llm = OpenAI(temperature=0, openai_api_key=__open_api_key)

    def __init__(self, max_tokens=500):
        """
        max_tokens is the max number of words in a response from the api
        """
        self.__embedding = Embedding()   # TODO: impliment embedding

    def query(self, query):
        rag_docs = self.__embedding.get_similar_documents(query)

        chain = load_qa_chain(self.open_llm, chain_type="stuff")
        out = chain.run(input_documents=rag_docs, question=query)
        return out

class Embedding(ABC):
    @abstractmethod
    def run_batch_embeddings_ingestion(self):
        pass

    def __load_chunked_xray_articles_csv(self, chunked_article_csv="RAG/datasets/xray_articles_chunked.csv"):
        return pd.read_csv(chunked_article_csv)

class Hugging_face_embedding(Embedding):
    def __init__(self, chunked_article_csv="RAG/datasets/xray_articles_chunked.csv"):
        self.chunked_article_csv = chunked_article_csv

    def run_batch_embeddings_ingestion(self, num_docs = None):
        df = self.load_chunked_xray_articles_csv(self.chunked_article_csv)
        if num_docs == None:
            new_df = df.copy()
        else :
            new_df = df.head(num_docs).copy()
        chunks = new_df['text'].tolist()
        # one call per document
        pub_ids = new_df['PMID'].tolist()

        # create a dictionary of pub_id -> list[chunk]
        pub_id_to_chunks = {}
        for i, pub_id in enumerate(pub_ids):
            if pub_id not in pub_id_to_chunks:
                pub_id_to_chunks[pub_id] = []
            pub_id_to_chunks[pub_id].append(chunks[i])

        # do one call per pub_id'
        embeddings_as_lists = []
        for pub_id in pub_id_to_chunks:
            embedding_result = None
                # Logic for HuggingFace embeddings
            embedding_result = self.embedding_model.embed_documents(pub_id_to_chunks[pub_id])
            embeddings_as_lists.extend([list(embedding) for embedding in embedding_result])
            print("Finished embedding pub_id: ", pub_id)

        new_df['embedding'] = embeddings_as_lists
        new_df.to_csv("RAG/datasets/xray_articles_with_embeddings2.csv", index=True)

class Openai_embedding(Embedding):
    def __init__(self, chunked_article_csv="RAG/datasets/xray_articles_chunked.csv"):
        self.chunked_article_csv = chunked_article_csv

    def run_batch_embeddings_ingestion(self, num_docs = None):
        df = self.load_chunked_xray_articles_csv(self.chunked_article_csv)
        if num_docs == None:
            new_df = df.copy()
        else :
            new_df = df.head(num_docs).copy()
        chunks = new_df['text'].tolist()
        # one call per document
        pub_ids = new_df['PMID'].tolist()

        # create a dictionary of pub_id -> list[chunk]
        pub_id_to_chunks = {}
        for i, pub_id in enumerate(pub_ids):
            if pub_id not in pub_id_to_chunks:
                pub_id_to_chunks[pub_id] = []
            pub_id_to_chunks[pub_id].append(chunks[i])

        # do one call per pub_id'
        embeddings_as_lists = []
        for pub_id in pub_id_to_chunks:
            embedding_result = None
                # Logic for HuggingFace embeddings
            embedding_result = self.embedding_open.aembed_documents(pub_id_to_chunks[pub_id])
            embeddings_as_lists.extend([list(embedding) for embedding in embedding_result])
            print("Finished embedding pub_id: ", pub_id)

        new_df['embedding'] = embeddings_as_lists
        new_df.to_csv("RAG/datasets/xray_articles_with_embeddings2.csv", index=True)

