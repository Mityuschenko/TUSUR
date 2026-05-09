
from datetime import datetime, timezone
import uuid  # Импортируем библиотеку для генерации уникальных ID

from dependencies import chroma, record_manager 
from schemas.document_upload import DocumentUpload 

from fastapi import APIRouter, Body 


from langchain_core.indexing import aindex
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 


router = APIRouter() 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators=["\n", "\n\n", "."],
)

@router.post(
    "/load/chroma", 
    description="Загрузка документа с добавлением родительского ID и даты в метаданные"
)
async def upload_document(document_input: DocumentUpload = Body()):
    # 1. Генерируем уникальный ID для родительского документа (строковый формат)
    parent_id = str(uuid.uuid4())
    
    # 2. Получаем текущее время
    current_time = datetime.now(timezone.utc).isoformat()
    
    # 3. Собираем метаданные: объединяем пользовательские данные, ID и время
    final_metadata = document_input.metadata.copy() if document_input.metadata else {}
    final_metadata["parent_doc_id"] = parent_id
    final_metadata["uploaded_at"] = current_time
    
    # 4. Нарезаем текст. Все чанки из этого запроса получат одинаковый parent_doc_id
    chunks: list[Document] = text_splitter.create_documents(
        texts=[document_input.content], 
        metadatas=[final_metadata]
    )

    out = await aindex(
        chunks, 
        record_manager=record_manager, 
        vector_store=chroma
    )
    
    # 5. Возвращаем статистику индексации вместе с сгенерированным parent_doc_id
    return {
        "index_status": out,
        "parent_doc_id": parent_id
    }
