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

class PubmedEmbedding:
    def __init__(self):
        dotenv.load_dotenv()
        self.co = Client(os.getenv('COHERE_API_KEY'))
        self.xray_articles = self.load_xray_articles()
        self.embedding_model = HuggingFaceEmbeddings("all-MiniLM-L6-v2")
    
    def load_xray_articles(self):
        with open("datasets/xray_articles.json", "r") as f:
            return json.load(f)
    
    def load_chunked_xray_articles(self):
        with open("datasets/xray_articles_chunked.json", "r") as f:
            return json.load(f)
    
    def convert_json_to_df(self, json_data=None):
        if json_data is None:
            df = pd.DataFrame(self.xray_articles)
        else:
            df = pd.DataFrame(json_data)
        return df
    def print_metadata(self):
        df = self.convert_json_to_df()
        print("Length of unprocessed dataset: ", len(df))
        print("Columns: ", df.columns)
        print("Sample row: ", df.iloc[0])
        df2 = pd.DataFrame(self.load_chunked_xray_articles())
        print("Length of chunked dataset: ", len(df2))
        print("Columns: ", df2.columns)
        print("Sample row: ", df2.iloc[0])

    def chunk_text(self, tokens, num_tokens=100):

        return [tokens[i:i+num_tokens] for i in range(0, len(tokens), num_tokens)]

    # Pardon my ghetto tokenization LOL
    def create_chunked_dataset(self, max_tokens=100):
        df = self.convert_json_to_df()
        chunked_dataset = []
        for i, row in df.iterrows():
            j = 0
            # fulltext is an array of strings
            full_text = " ".join(row['FullText']).split(" ")
            # tokenize 

            for chunk in self.chunk_text(full_text, max_tokens):
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
        with open("datasets/xray_articles_chunked.json", "w") as f:
            json.dump(chunked_dataset, f)

    def convert_json_to_csv(self, json_data=None):
        df = self.convert_json_to_df(json_data)
        df.to_csv("datasets/xray_articles.csv")

    def run_batch_embeddings_ingestion(self):
        json_data = self.load_chunked_xray_articles()
        df = self.convert_json_to_df(json_data)
        new_df = df.copy()
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
            embedding_result = self.co.embed(pub_id_to_chunks[pub_id], model="embed-english-v3.0", input_type="search_query")
            embeddings_as_lists.extend([list(embedding) for embedding in embedding_result.embeddings])
            # sleep for 1 second to avoid rate limiting
            time.sleep(1)
            print("Finished embedding pub_id: ", pub_id)
        
        new_df['embedding'] = embeddings_as_lists
        new_df.to_csv("datasets/xray_articles_with_embeddings.csv")
        
    def retrieve_embeddings(self):
        df = pd.read_csv("datasets/xray_articles_with_embeddings.csv")
        return df['embedding'].tolist()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pubmed Embedding Tool")
    parser.add_argument("-r", "--run_batch", action="store_true", help="Run batch embeddings ingestion")
    parser.add_argument("-e", "--retrieve_embeddings", action="store_true", help="Retrieve embeddings")
    parser.add_argument("-p", "--print_metadata", action="store_true", help="Print metadata")
    parser.add_argument("-c", "--create_chunked_dataset", action="store_true", help="Create chunked dataset")
    args = parser.parse_args()

    pe = PubmedEmbedding()
    if args.run_batch:
        pe.run_batch_embeddings_ingestion()
    if args.retrieve_embeddings:
        embeddings = pe.retrieve_embeddings()
        print(embeddings)
    if args.print_metadata:
        pe.print_metadata()
    if args.create_chunked_dataset:
        pe.create_chunked_dataset()
    if args.similarity_search:
        pass

    # If no arguments are provided, display the help message
    if not any(vars(args).values()):
        parser.print_help()



