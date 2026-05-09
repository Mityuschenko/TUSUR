
from chroma_async import AsyncChroma
from constants import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDINGS_MODEL,
    LLM_MODEL,
    RECORD_MANAGER,
)
from langchain_classic.indexes import SQLRecordManager
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Инициализация эмбеддингов
embedding_model = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

# Подключение к Chroma
chroma = AsyncChroma(
    collection_name=CHROMA_COLLECTION,
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model,
)

# Инициализация менеджера записей для отслеживания состояния документов
record_manager = SQLRecordManager(
    namespace="Stardew", db_url=RECORD_MANAGER, async_mode=True
)

# Инициализация LLM
llm = ChatOllama(model=LLM_MODEL)
