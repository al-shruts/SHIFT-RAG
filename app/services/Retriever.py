from typing import List
from langchain.docstore.document import Document

from app.config import config


class Retriever:
    """
    Класс Retriever предназначен для поиска и извлечения контекста из документов с использованием векторного хранилища.

    Attributes:
        indexer (Any): Индексатор для создания и загрузки векторного хранилища.
        embeddings (Any): Инструмент для получения векторных представлений данных.
        index_path (Path): Путь к файлу с индексом.
        k_search (int): Количество результатов, возвращаемых при поиске.
        vectorstore (Any): Векторное хранилище для хранения документов.
        retriever (Any): Инструмент для поиска по векторному хранилищу.

    Args:
        indexer (Any): Индексатор для создания и загрузки векторного хранилища.
        embeddings (Any): Инструмент для получения векторных представлений данных.
        index_path (str): Путь к файлу с индексом.
        k_search (int, optional): Количество результатов, возвращаемых при поиске. По умолчанию 5.
    """

    def __init__(self, indexer, embeddings, index_path: str, k_search: int = 5):
        self.indexer = indexer
        self.embeddings = embeddings
        self.index_path = config.PROJECT_DIR / index_path
        self.k_search = k_search
        self.vectorstore = None
        self.retriever = None

    def setup(self):
        """
        Настраивает инструмент для поиска по векторному хранилищу.
        """
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': self.k_search})

    def build(self, documents: List[Document]):
        """
        Создает векторное хранилище из списка документов и сохраняет его.

        Args:
            documents (List[Document]): Список документов для создания векторного хранилища.
        """
        self.vectorstore = self.indexer.from_documents(documents, self.embeddings)
        self.save()

    def load(self):
        """
        Загружает векторное хранилище из локального файла.
        """
        try:
            self.vectorstore = self.indexer.load_local(folder_path=self.index_path, embeddings=self.embeddings,
                                                       allow_dangerous_deserialization=True)
        except RuntimeError as exc:
            print(f"Index path `{self.index_path}` does not exist.")

    def save(self):
        """
        Сохраняет векторное хранилище в локальный файл.
        """
        self.vectorstore.save_local(folder_path=self.index_path)

    def set(self, documents: List[Document]):
        """
        Добавляет новые документы в существующее векторное хранилище и сохраняет его.

        Args:
            documents (List[Document]): Список новых документов для добавления.
        """
        vectorstore = self.indexer.from_documents(documents, self.embeddings)
        if self.vectorstore:
            self.vectorstore.merge(vectorstore)
        else:
            self.vectorstore = vectorstore

        self.save()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': self.k_search})

    def get(self, question: str, k: int = None) -> List[Document]:
        """
        Извлекает документы из векторного хранилища на основе заданного вопроса.

        Args:
            question (str): Вопрос пользователя.
            k (int, optional): Количество результатов поиска. По умолчанию None.

        Returns:
            List[Document]: Список найденных документов.
        """
        if k:
            self.retriever.search_kwargs['k'] = k

        return self.retriever.invoke(question)
