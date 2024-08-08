from typing import Any, Dict, List
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Config
from .Reader import Reader
from .Retriever import Retriever
from .MetricEvaluator import MetricEvaluator


class QAService:
    """
    Класс для предоставления услуг вопросов и ответов с использованием извлечения информации и языковых моделей.

    Attributes:
        metric_evaluator (MetricEvaluator): Объект для оценки метрики схожести.
        cache (Retriever): Объект для извлечения документов из кеша.
        retriever (Retriever): Объект для извлечения документов из основного индекса.
        llm (Any): Языковая модель для генерации ответов.
        threshold (float): Порог схожести для фильтрации документов.

    Args:
        indexer (Any): Объект для индексирования данных.
        embeddings (Any): Объект для создания эмбеддингов.
        index_path (str): Путь к индексу данных.
        data_path (str): Путь к данным для индексации.
        llm (Any): Языковая модель.
        k_search (int, optional): Количество результатов поиска. По умолчанию 5.
        threshold (float, optional): Порог схожести для фильтрации документов. По умолчанию 0.5.
    """

    def __init__(self,
                 indexer: Any,
                 embeddings: Any,
                 index_path: str,
                 data_path: str,
                 llm: Any,
                 k_search: int = 5,
                 threshold: float = 0.5):
        self.metric_evaluator = MetricEvaluator(embeddings)
        self.cache = Retriever(indexer, embeddings, index_path + '/cache', k_search)
        self.retriever = Retriever(indexer, embeddings, index_path + '/index', k_search)
        self.llm = llm
        self.threshold = threshold

        # Cache load and setup
        self.cache.load()
        if self.cache.vectorstore:
            self.cache.setup()

        # Retriever load and setup
        self.retriever.load()
        if self.retriever.vectorstore is None:
            reader = Reader()
            documents = reader.read_documents(data_path)
            self.retriever.build(documents)

        self.retriever.setup()

    def detect_intent(self, que: str) -> bool:
        """
        Определяет намерение пользователя на основе заданного вопроса.

        Args:
            que (str): Вопрос пользователя.

        Returns:
            bool: True, если намерение не начинается с 'да', иначе False.
        """

        intent = self.llm.invoke(Config().PROMPTS['intent'].format(question=que)).content.lower()
        return False if intent.startswith('да') else True

    def filter_based_on_metric(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Фильтрует документы на основе метрики схожести с заданным вопросом.

        Args:
            question (str): Вопрос пользователя.
            documents (List[Document]): Список документов для фильтрации.

        Returns:
            List[Document]: Отфильтрованный список документов.
        """
        result = []

        for document in documents:
            score = self.metric_evaluator.evaluate(question, document.page_content)
            if score > self.threshold:
                result.append(document)

        return result

    def get_cached_answer(self, question: str, k: int = None):
        """
        Извлекает ответ из кеша на основе заданного вопроса.

        Args:
            question (str): Вопрос пользователя.
            k (int, optional): Количество результатов поиска. По умолчанию None.

        Returns:
            List[str] | None: Список ответов из кеша или None, если ответы не найдены.
        """
        documents = self.cache.get(question, k)

        if documents:
            documents = self.filter_based_on_metric(question, documents)
            if not documents:
                return None

        return [document.metadata.get('answer') for document in documents]

    def get_llm_answer(self, question: str):
        """
        Генерирует ответ с использованием языковой модели на основе заданного вопроса.

        Args:
            question (str): Вопрос пользователя.

        Returns:
            str: Сгенерированный ответ.
        """
        context = ''
        if self.detect_intent(question):
            documents = self.retriever.get(question)
            context = "\n\n".join([document.page_content for document in documents])

        messages = [
            SystemMessage(content=Config().PROMPTS['system'].format(context=context)),
            HumanMessage(content=Config().PROMPTS['user'].format(question=question)),
        ]

        result = ''
        for response in self.llm.stream(messages):
            result += response.content or ''
        result = result.strip('Ответ:')

        return result

    def set_cache(self, qa_pairs: Dict[str, str]):
        """
        Устанавливает кеш вопросов и ответов.

        Args:
            qa_pairs (Dict[str, str]): Словарь пар вопросов и ответов.
        """
        documents = []
        for question, answer in qa_pairs.items():
            document = Document(
                page_content=question,
                metadata={"answer": answer}
            )
            documents.append(document)

        self.cache.set(documents)
