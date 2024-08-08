from fastapi import APIRouter, Query, Request
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .services.LLM import DummyLLM
from .services.QAService import QAService

router = APIRouter()
llm = DummyLLM()
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
qa_service = QAService(FAISS, embeddings, 'index', 'data', llm)


@router.get('/ask')
async def ask(question: str = Query(..., description="User's question")):
    response = False
    try:
        response = qa_service.get_cached_answer(question, 1)
        if not response:
            response = qa_service.get_llm_answer(question)
    except Exception as exc:
        print(exc)
    finally:
        return {"response": response}


@router.get('/ask-llm')
async def ask_llm(question: str = Query(..., description="User's question")):
    response = False
    try:
        response = qa_service.get_llm_answer(question)
    except Exception as exc:
        print(exc)
    finally:
        return {"response": response}


@router.post('/set-cache')
async def set_cache(request: Request):
    response = False
    try:
        data = await request.json()
        response = qa_service.set_cache(data)
    except Exception as exc:
        print(exc)
    finally:
        return {"response": response}
