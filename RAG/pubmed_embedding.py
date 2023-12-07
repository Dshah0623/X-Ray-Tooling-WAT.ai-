import json
from cohere import Client
import os
import pandas as pd
import dotenv

class PubmedEmbedding:
    def __init__(self):
        dotenv.load_dotenv()
        self.co = Client(os.getenv('COHERE_API_KEY'))
        self.xray_articles = self.load_xray_articles()
    
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
    

        
if __name__ == "__main__":
    pe = PubmedEmbedding()
    pe.create_chunked_dataset()
    pe.print_metadata()
    pe.convert_json_to_csv(pe.load_chunked_xray_articles())