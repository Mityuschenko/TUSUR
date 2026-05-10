
from fastapi import FastAPI
from routers.document_loader import router  # Импорт роутера из файла, где вы писали код

app = FastAPI(title="Документация Chroma Indexer")

# Подключаем роутер
app.include_router(router)
