import os
import re
from typing import List
from datetime import datetime
from langchain.docstore.document import Document

from app.config import config


class Reader:
    """
    Класс Reader предназначен для чтения документов из файлов, извлечения их содержимого и метаданных.

    Attributes:
        metadata_pattern (re.Pattern): Регулярное выражение для поиска метаданных в тексте.
    """

    def __init__(self):
        """
        Инициализация класса Reader.
        """
        self.metadata_pattern = re.compile(
            r' Metadata\s*link: (https?://[^\s]+)\s*date: (\d{2}-\d{4})', re.DOTALL
        )

    def extract_metadata(self, text: str) -> tuple:
        """
        Извлечение метаданных из текста документа.

        Ищет в тексте метаданные, такие как ссылка и дата, и возвращает очищенный текст и словарь метаданных.

        Args:
            text (str): Текст документа, из которого необходимо извлечь метаданные.

        Returns:
            tuple: Кортеж, содержащий очищенный текст и словарь с метаданными.
                   Если метаданные не найдены, возвращается исходный текст и пустой словарь.
        """
        match = self.metadata_pattern.search(text)
        if match:
            metadata_text = match.group(0)
            clean_text = text.replace(metadata_text, '').strip()
            link = match.group(1).strip()
            date_str = match.group(2).strip()
            try:
                date = datetime.strptime(date_str, '%d-%m-%Y')
            except ValueError:
                date = None
            metadata = {
                'link': link,
                'date': date
            }
            return clean_text, metadata
        return text, {}

    def read_documents(self, directory: str) -> List[Document]:
        """
        Чтение документов из указанной директории и создание списка объектов Document.

        Считывает файлы с расширением '.md' из указанной директории, извлекает их содержимое и метаданные,
        и возвращает список объектов Document.

        Args:
            directory (str): Путь к директории, содержащей документы.

        Returns:
            List[Document]: Список объектов Document, созданных на основе файлов в указанной директории.
        """
        documents = []
        for root, _, files in os.walk(config.PROJECT_DIR / directory):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        text, metadata = self.extract_metadata(content)
                        documents.append(Document(page_content=text, metadata=metadata))

        return documents
