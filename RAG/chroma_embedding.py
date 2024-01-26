import os
import json
import dotenv
import argparse
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from embedding import Embedding


class ChromaEmbedding(Embedding):
    """
    A class for handling embedding operations and Chroma database interactions.

    This class provides methods to set up embedding models, process x-ray articles,
    create, populate, and manage a Chroma vector database.

    Attributes:
        __open_key (str): The API key for OpenAI.
        __embeddings_hugging (HuggingFaceEmbeddings): HuggingFace embedding model.
        __embedding_open (OpenAIEmbeddings): OpenAI embedding model.
        __persist_chroma_directory (str): Directory for persisting Chroma database.
        __xray_articles (list): Loaded x-ray articles.
        __xray_chunked_articles (list): Chunked x-ray articles.
        __embedding_in_use (object): The embedding model currently in use.
        __chroma_db (Chroma): Chroma vector database instance.
    """
    dotenv.load_dotenv()
    __open_key = os.getenv('OPENAI_API_KEY')
    __embeddings_hugging = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    __embedding_open = OpenAIEmbeddings(openai_api_key=__open_key)
    __persist_chroma_directory = 'db'

    def __init__(
            self,
            use_open_ai=False,
            num_matches=5,
            dataset_path="RAG/datasets/"
    ) -> None:
        self.__xray_articles = self.__load_xray_articles()
        self.__xray_chunked_articles = self.__chunk_documents(
            self.__xray_articles)
        self.__embedding_in_use = None
        self.__chroma_db = None
        self.__num_matches = num_matches
        self.__processed_articles_path = dataset_path + "xray_articles_processed.json"
        self.__articles_path = dataset_path + "xray_articles.json"
        self.set_embedding_model(use_open_ai)
        self.load_chroma_db()
        self.create_and_populate_chroma()

    def set_embedding_model(self, use_open_ai=False) -> None:
        """
        Sets the embedding model to be used based on the user's choice.

        Args:
            use_hugging_face (bool): If True, sets HuggingFace embeddings.
            use_open_ai (bool): If True, sets OpenAI embeddings.

        Raises:
            ValueError: If no embedding model is selected.
        """
        if use_open_ai:
            self.__embedding_in_use = self.__embedding_open
        else:
            self.__embedding_in_use = self.__embeddings_hugging

    def __load_and_chunk_articles(self) -> object:
        docs = self.__load_xray_articles()
        self.__xray_chunked_articles = self.__chunk_documents(docs)

    def __process_json(self) -> object:
        # Load the original JSON
        with open(self.__articles_path, "r") as file:
            data = json.load(file)

        # Process each document
        for doc in data:
            doc['Authors'] = ' , '.join(doc['Authors'])
            doc['FullText'] = ' , '.join(doc['FullText'])

        # Save the processed JSON
        with open(self.__processed_articles_path, "w") as file:
            json.dump(data, file, indent=4)

    def __load_xray_articles(self) -> object:
        loader = JSONLoader(
            file_path=self.__processed_articles_path,
            jq_schema='.[].FullText',
            text_content=True)

        return loader.load()

    def __chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200) -> object:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    def create_and_populate_chroma(self) -> None:
        """
        Creates a Chroma database from chunked x-ray articles and populates it with embeddings.
        """
        vector_db = Chroma.from_documents(documents=self.__xray_chunked_articles,
                                          embedding=self.__embedding_in_use,
                                          persist_directory=self.__persist_chroma_directory)

        vector_db.persist()

    def __create_ids(self) -> list:
        ids = [str(i) for i in range(1, len(self.__xray_chunked_articles) + 1)]

    def load_chroma_db(self) -> None:
        """
        Loads the Chroma database from the persistent storage.
        """
        vector_db = Chroma(persist_directory=self.__persist_chroma_directory,
                           embedding_function=self.embedding_in_use, ids=self.__create_ids())

        self.__chroma_db = vector_db

    def get_similar_documents(self, query) -> object:
        """
        Retrieves documents from the Chroma database based on a given query.

        Args:
            query (str): The query string for document retrieval.
            search_kwargs (int, optional): Additional search parameters. Defaults to 2.

        Returns:
            object: Retrieved documents from the Chroma database.
        """

        # TODO restrict return doc amount to self.__num_matches
        docs = self.__chroma_db.similarity_search(query)
        return docs

    def reupload_to_chroma(self) -> None:
        """
        Clears the current Chroma database and re-uploads chunked articles.
        """
        self.clear_chroma()
        self.__load_and_chunk_articles()
        self.__chroma_db = Chroma.from_documents(
            self.__xray_chunked_articles, ids=self.__create_ids())

    def clear_chroma(self) -> None:
        """
        Clears documents from the Chroma database.
        """
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

    def destroy(self):
        self.clear_chroma()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma Embedding Tool")
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # Subparser for set_embedding_model
    parser_set_model = subparsers.add_parser(
        'set_model', help='Set the embedding model')
    parser_set_model.add_argument(
        '-oi', '--openai', action='store_true', help='Use OpenAI embeddings instead of default huggingface')

    # Subparser for build_chroma
    parser_build_chroma = subparsers.add_parser(
        'build_chroma', help='Create and populate Chroma DB')

    # Subparser for load_chroma
    parser_load_chroma = subparsers.add_parser(
        'load_chroma', help='Load Chroma DB')

    # Subparser for retrieve_from_query
    parser_retrieve = subparsers.add_parser(
        'retrieve_from_query', help='Retrieve documents from Chroma based on query')
    parser_retrieve.add_argument(
        'query', type=str, help='Query for document retrieval')

    # Subparser for reupload
    parser_reupload = subparsers.add_parser(
        'reupload', help='Reupload documents to Chroma')

    # Subparser for clear_db
    parser_clear_db = subparsers.add_parser(
        'clear_db', help='Clear Chroma DB')

    args = parser.parse_args()

    chroma = ChromaEmbedding()

    if args.command == 'set_model':
        chroma.set_embedding_model(use_open_ai=args.openai)
    elif args.command == 'build_chroma':
        chroma.create_and_populate_chroma()
    elif args.command == 'load_chroma':
        chroma.load_chroma_db()
    elif args.command == 'get_similar_documents':
        print(chroma.get_similar_documents(args.query))
    elif args.command == 'reupload':
        chroma.reupload_to_chroma()
    elif args.command == 'clear_db':
        chroma.clear_chroma()
    else:
        parser.print_help()
