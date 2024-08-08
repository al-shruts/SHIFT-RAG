import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
    PROMPTS: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('PROMPTS', mode='before')
    def load_texts(cls, v):
        if not v:
            file_path = Path(__file__).resolve().parent.parent / 'prompts.json'
            if not file_path.exists():
                print(f"File does not exist: {file_path}")
                return {}
            try:
                with file_path.open('r', encoding='utf-8') as file:
                    return json.load(file)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return {}
        return v


config = Config()
