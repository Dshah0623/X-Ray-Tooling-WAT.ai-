import json
from cohere import Client
import os
import errno
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
from embedding import Embedding


class IndexEmbedding(Embedding):
    __open_key = os.getenv('OPENAI_API_KEY')
    __embeddings_hugging = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    __embedding_open = OpenAIEmbeddings(openai_api_key=__open_key)

    def __init__(
            self, 
            use_openai = False,
            chunking_max_tokens=100,
            num_matches=5,
            dataset_path="RAG/datasets/"
            ) -> None:
        self.__articles_path = dataset_path + "xray_articles.json"
        self.__embedding_path = dataset_path + "xray_articles_with_embeddings2.csv"
        self.__index_path = dataset_path + "xray_articles_vector_index.pkl"
        self.__chunked_articles_csv_path = dataset_path + "xray_articles_chunked.csv"
        self.__chunked_articles_json_path = dataset_path + "xray_articles_chunked.json"
        self.__chunking_max_tokens = chunking_max_tokens
        self.__num_matches = num_matches
        self.__use_openai = use_openai
        self.__xray_chunked_articles = self.__load_chunked_xray_articles_csv()

    def __load_xray_articles(self, path):
        if not os.path.exists(self.__index_path):
            raise ValueError("Articled file not found!")
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
        
    def __load_chunked_xray_articles_csv(self):
        if not os.path.exists(self.__chunked_articles_csv_path):
            self.__create_chunked_dataset(self.__chunking_max_tokens)
        return pd.read_csv(self.__chunked_articles_csv_path)
        
    def __convert_json_to_df(self, json_data=None):
        if json_data is None:
            df = pd.DataFrame(
                self.__load_xray_articles(self.__articles_path)
                )
        else:
            df = pd.DataFrame(json_data)
        return df
    
    def __save_json_as_csv(self, json_path, save_path):
        df = None
        with open(json_path, "r", encoding='utf-8') as f:
            df = json.load(f)
        df.to_csv(save_path)
    
    def __chunk_text(self, tokens):
        return [tokens[i:i+self.__num_tokens] for i in range(0, len(tokens), self.__num_tokens)]

    def __create_chunked_dataset(self):
        df = self.__convert_json_to_df()
        chunked_dataset = []
        for _, row in df.iterrows():
            j = 0
            # fulltext is an array of strings
            full_text = " ".join(row['FullText']).split(" ")
            
            # tokenize
            for chunk in self.__chunk_text(full_text):
                chunk_text = " ".join(chunk)
                chunked_dataset.append({
                    "title": row['Title'],
                    "authors": row['Authors'],
                    "publication_date": row['PublicationDate'],
                    "PMID": row['PMID'],
                    "text": chunk_text,
                    "chunk_num": j
                })
                j += 1
        # save as json
        with open(self.__chunked_articles_json_path, "w") as f:
            json.dump(chunked_dataset, f)

        self.__save_json_as_csv(self.__chunked_articles_json_path, self.__chunked_articles_csv_path)

    def __run_batch_embeddings_ingestion(self):
        df = self.__xray_chunked_articles
        new_df = df.head(5).copy()
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
            if self.use_openai:
                # Logic for OpenAI embeddings
                embedding_result = self.__embedding_open.aembed_documents(
                    pub_id_to_chunks[pub_id])
            else:
                # Logic for HuggingFace embeddings
                embedding_result = self.__embeddings_hugging.embed_documents(
                    pub_id_to_chunks[pub_id])
            embeddings_as_lists.extend([list(embedding)
                                       for embedding in embedding_result])

        new_df['embedding'] = embeddings_as_lists
        new_df.to_csv(self.__embedding_path, index=True)

    def __cosine_similarity(self, v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)

    def __build_vector_index(self):
        if not os.path.exists(self.__embedding_path):
            self.__run_batch_embeddings_ingestion()
        df = pd.read_csv(self.__embedding_path)
        index = [(row['index'], row['embedding'], row['text'])
                 for _, row in df.iterrows()]
        # make embeddings into list of floats
        index = [(row[0], [float(x) for x in row[1][1:-1].split(",")], row[2])
                 for row in index]
        # save as a pickle
        with open(self.__index_path, "wb") as f:
            pickle.dump(index, f)

    def __retrieve_vector_index(self):
        """
        returns list and latency
        """
        list_of_embeddings = []
        start = time.time()
        if not os.path.exists(self.__index_path):
            self.__build_vector_index()
        with open(self.__index_path, "rb") as f:
            list_of_embeddings = pickle.load(f)

        return list_of_embeddings, time.time() - start
    
    def __silent_remove(path):
        try:
            os.remove(path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def __clean_directory(self):
        self.__silent_remove(self.__embedding_path)
        self.__silent_remove(self.__index_path)
        self.__silent_remove(self.__chunked_articles_csv_path)
        self.__silent_remove(self.__chunked_articles_json_path)

    def get_similar_documents(self, query):
        embeddings, _ = self.__retrieve_vector_index()
        if self.__use_openai:
            # Logic for OpenAI embeddings
            embedding_result = self.__embedding_open.aembed_query(query)
        else:
            # Logic for HuggingFace embeddings
            embedding_result = self.__embeddings_hugging.embed_query(query)
        similarity_scores = []
        for (i, embedding, chunk) in embeddings:
            similarity_scores.append(
                (self.__cosine_similarity(embedding_result, embedding), i, chunk))
        sorted_similarity_scores = sorted(
            similarity_scores, key=lambda x: x[0], reverse=True)
        return sorted_similarity_scores[:self.__num_matches]
    
    def destroy(self):
        self.__clean_directory()

# if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Vector Index Embedding Tool")
    # subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # # Subparser for build_chroma
    # parser_build_index = subparsers.add_parser(
    #     'build_index', help='Create and populate index DB')
    # parser_build_index.add_argument(
    #     '-oi', '--openai', action='store_true', help='Use OpenAI embeddings instead of the default huggingface')

    # # Subparser for retrieve_from_query
    # parser_retrieve = subparsers.add_parser(
    #     'retrieve_from_query', help='Retrieve documents from Chroma based on query')
    # parser_retrieve.add_argument(
    #     'query', type=str, help='Query for document retrieval')

    # # Subparser for reupload
    # parser_reumake_index = subparsers.add_parser(
    #     'remake', help='Remake the vector index')

    # # Subparser for clear_db
    # parser_delete_index = subparsers.add_parser(
    #     'delete_index', help='Delete the vector index')

    # args = parser.parse_args()

    # embedding = IndexEmbedding()

    # if args.command == 'build_index':
    #     embedding.create_and_populate_chroma(
    #         use_openai=args.openai
    #     )
    # elif args.command == 'retrieve_from_query':
    #     print(chroma.retrieve_from_chroma(args.query))
    # elif args.command == 'reupload':
    #     chroma.reupload_to_chroma()
    # elif args.command == 'clear_db':
    #     chroma.clear_chroma()
    # else:
    #     parser.print_help()