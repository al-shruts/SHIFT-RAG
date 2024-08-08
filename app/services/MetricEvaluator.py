from typing import Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim


class MetricEvaluator:
    """
    Класс для оценки схожести строк с использованием как лексического, так и семантического анализа.

    Attributes:
        embeddings (Any): Объект для создания эмбеддингов текста.

    Args:
        embeddings (Any): Объект для создания эмбеддингов текста.
    """

    def __init__(self, embeddings: Any):
        self.embeddings = embeddings

    def lexical_cosine_similarity(self, str1: str, str2: str) -> float:
        """
        Вычисляет лексическое косинусное сходство между двумя строками.

        Args:
            str1 (str): Первая строка.
            str2 (str): Вторая строка.

        Returns:
            float: Значение косинусного сходства между строками.
        """
        vectorizer = CountVectorizer().fit_transform([str1, str2])
        vectors = vectorizer.toarray()
        return cosine_sim(vectors)[0][1]

    def semantic_cosine_similarity(self, str1: str, str2: str) -> float:
        """
        Вычисляет семантическое косинусное сходство между двумя строками.

        Args:
            str1 (str): Первая строка.
            str2 (str): Вторая строка.

        Returns:
            float: Значение косинусного сходства между строками на основе эмбеддингов.
        """
        embeddings1 = self.get_embeddings(str1)
        embeddings2 = self.get_embeddings(str2)
        return cosine_sim([embeddings1], [embeddings2])[0][0]

    def get_embeddings(self, text: str) -> list:
        """
        Получает эмбеддинги для заданного текста.

        Args:
            text (str): Текст для создания эмбеддингов.

        Returns:
            list: Список эмбеддингов для текста.
        """
        return self.embeddings.embed_query(text)

    def evaluate(self, str1: str, str2: str) -> float:
        """
        Оценивает схожесть двух строк на основе комбинации лексического и семантического анализа.

        Args:
            str1 (str): Первая строка.
            str2 (str): Вторая строка.

        Returns:
            float: Комбинированное значение схожести строк.
        """
        lexical_score = self.lexical_cosine_similarity(str1, str2)
        semantic_score = self.semantic_cosine_similarity(str1, str2)

        # Взвешенная комбинация лексического и семантического сходства
        combined_score = 0.5 * lexical_score + 0.5 * semantic_score

        return combined_score
