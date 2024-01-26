from abc import ABC, abstractmethod
import pandas as pd

class Embedding(ABC):
    @abstractmethod
    def get_similar_documents(self, query, search_kwargs=5):
        pass

    @abstractmethod
    def destroy(self):
        pass