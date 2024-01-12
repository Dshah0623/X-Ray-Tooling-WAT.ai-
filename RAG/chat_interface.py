from cohere import Client
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import uuid

class Chat(ABC):
    @abstractmethod
    def query(self, query):
        """
        run query through the chat to get a response with rag
        """
        pass

# TODO: create the child classes... not sure where we're at for this + blocked by not being able to run the embedding on windows
    # see pubmed_embedding.py for inspiration
class Embedding(ABC):
    @abstractmethod
    def get_similar_documents(self, query):
        pass

class Cohere(Chat):
    def __init__(self, max_tokens=500):
        """
        max_tokens is the max number of words in a response from the api
        """
        load_dotenv()
        cohere_api_key = os.getenv('COHERE_API_KEY')
        self.co = Client(cohere_api_key)
        self.conversation_id = str(uuid.uuid4())
        self.max_tokens = max_tokens
        self.embedding = Embedding()   # TODO: impliment embedding

    def query(self, query):
        rag_docs = self.embedding.get_similar_documents(query)

        response = self.co.chat(
            message=query,
            documents=rag_docs,
            conversation_id=self.conversation_id,
            max_tokens=self.max_tokens,
        )
        return response.text
    
class OpenAI(Chat):
    def __init__(self, max_tokens=500):
        """
        max_tokens is the max number of words in a response from the api
        """
        load_dotenv()
        open_api_key = os.getenv('OPENAI_API_KEY')
        self.open_llm = OpenAI(
            temperature=0, openai_api_key=open_api_key)
        self.embedding = Embedding()   # TODO: impliment embedding

    def query(self, query):
        rag_docs = self.embedding.get_similar_documents(query)

        chain = load_qa_chain(self.open_llm, chain_type="stuff")
        out = chain.run(input_documents=rag_docs, question=query)
        return out
