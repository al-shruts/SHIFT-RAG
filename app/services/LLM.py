from abc import ABC, abstractmethod


# TODO: Реализовать класс выбранной LLM
class LLM(ABC):
    """
    Абстрактный класс для LLM, определяющий интерфейс для взаимодействия с моделью.

    Методы:
        invoke(question): Выполняет запрос к модели с заданным вопросом.
        stream(messages): Потоковое получение ответов от модели на основе заданных сообщений.
    """

    @abstractmethod
    def invoke(self, question):
        """
        Выполняет запрос к модели с заданным вопросом.

        Args:
            question (str): Вопрос, который требуется задать модели.

        Returns:
            Response: Объект ответа модели.
        """
        pass

    @abstractmethod
    def stream(self, messages):
        """
        Потоковое получение ответов от модели на основе заданных сообщений.

        Args:
            messages (list): Список сообщений для взаимодействия с моделью.

        Returns:
            list: Список объектов ответов модели.
        """
        pass


class DummyLLM(LLM):
    """
    DummyLLM - пример реализации абстрактного класса LLM, возвращающий фиксированные ответы.

    Методы:
        invoke(question): Возвращает фиктивный ответ на заданный вопрос.
        stream(messages): Возвращает список фиктивных ответов на основе заданных сообщений.
    """

    def invoke(self, question):
        """
        Возвращает фиктивный ответ на заданный вопрос.

        Args:
            question (str): Вопрос, который требуется задать модели.

        Returns:
            DummyResponse: Объект фиктивного ответа с контентом 'да'.
        """

        class DummyResponse:
            def __init__(self, content):
                self.content = content

        return DummyResponse('да')

    def stream(self, messages):
        """
        Возвращает список фиктивных ответов на основе заданных сообщений.

        Args:
            messages (list): Список сообщений для взаимодействия с моделью.

        Returns:
            list: Список объектов фиктивных ответов.
        """

        class DummyResponse:
            def __init__(self, content):
                self.content = content

        return [DummyResponse("Это пример ответа от модели. Ответ:")]
