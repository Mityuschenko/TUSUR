
import hashlib  # Импортируем стандартную библиотеку для генерации хэшей

from dependencies import chroma, record_manager 
from schemas.document_upload import DocumentUpload 

from fastapi import APIRouter, Body, Query  # Query нужен для необязательных параметров фильтрации 


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
    description="Загрузка документа с детерминированным ID для защиты от дубликатов"
)
async def upload_document(document_input: DocumentUpload = Body()):
    # 1. Генерируем стабильный parent_doc_id на основе хэша текста (SHA-256)
    # Для одного и того же текста этот ID всегда будет одинаковым
    text_hash = hashlib.sha256(document_input.content.encode("utf-8")).hexdigest()
    parent_id = text_hash

    # 2. Формируем метаданные, которые будут уникальны для этого контента
    final_metadata = document_input.metadata.copy() if document_input.metadata else {}
    final_metadata["parent_doc_id"] = parent_id
    
    # 3. Нарезаем текст на чанки
    chunks: list[Document] = text_splitter.create_documents(
        texts=[document_input.content], 
        metadatas=[final_metadata]
    )

    # 4. Передаем данные в aindex. Теперь при повторном запросе хэши чанков сойдутся,
    # и record_manager заблокирует дублирование в Chroma
    out = await aindex(
        chunks, 
        record_manager=record_manager, 
        vector_store=chroma
    )
    
    return {
        "index_status": out,
        "parent_doc_id": parent_id
    }

@router.get(
    "/get/documents", 
    description="Получить список всех чанков и их метаданных, записанных в Chroma"
)
async def get_stored_documents(
    parent_doc_id: str | None = Query(default=None, description="Фильтр по ID родительского документа"),
    limit: int = Query(default=10, description="Количество выводимых чанков")
):
    try:
        # 1. Формируем фильтр, если клиент передал parent_doc_id
        where_filter = {"parent_doc_id": parent_doc_id} if parent_doc_id else None
        
        # 2. Делаем прямой запрос к коллекции Chroma через метод .get()
        # Извлекаем тексты (documents) и метаданные (metadatas)
        result = chroma.get(
            where=where_filter,
            limit=limit,
            include=["documents", "metadatas"]  # Можно добавить "embeddings", если нужны сами векторы-числа
        )
        
        # 3. Форматируем ответ для удобного чтения на фронтенде или в Swagger
        formatted_chunks = []
        if result and "documents" in result:
            for idx in range(len(result["ids"])):
                formatted_chunks.append({
                    "chunk_id": result["ids"][idx],
                    "text": result["documents"][idx],
                    "metadata": result["metadatas"][idx] if result["metadatas"] else {}
                })
        
        return {
            "total_fetched": len(formatted_chunks),
            "chunks": formatted_chunks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения из Chroma: {str(e)}")
