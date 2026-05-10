
from fastapi import FastAPI
from routers.rag import router  # Импорт роутера из файла

app = FastAPI(title="Документация Chroma Indexer")

# Подключаем роутер
app.include_router(router)
