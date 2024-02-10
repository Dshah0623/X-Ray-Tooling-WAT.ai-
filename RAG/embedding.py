from abc import ABC, abstractmethod

class Embedding(ABC):
    @abstractmethod
    def get_similar_documents(self, query) -> object:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass