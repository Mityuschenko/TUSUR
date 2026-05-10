
# Импортируются готовые объекты векторной базы данных (chroma) и языковой модели (llm)
from dependencies import chroma, llm 

# Импортируется текстовый шаблон системного промпта для RAG из локального модуля
from prompts.rag_prompt import template 

# Импортируется Pydantic-схема для валидации входящих поисковых запросов пользователей
from schemas.search_query import SearchQuery 


# Импортируются инструменты FastAPI для создания маршрутов и явного определения тела запроса
from fastapi import APIRouter, Body 


# Импортируется цепочка для объединения найденных документов и их передачи в prompt
from langchain.chains.combine_documents import create_stuff_documents_chain 

# Импортируется главная цепочка для связывания ретривера (поисковика) и цепочки синтеза ответа
from langchain.chains.retrieval import create_retrieval_chain 

# Импортируется класс для создания структурированных шаблонов prompt-инструкций
from langchain_core.prompts import ChatPromptTemplate 


# Создается изолированный обработчик маршрутов (роутер) для интеграции в FastAPI приложение
router = APIRouter() 

# Объявляется асинхронный POST-метод для эндпоинта "/search/rag"
@router.post("/search/rag") 
async def rag(search_query: SearchQuery = Body()): # На вход ожидаются валидные данные из тела HTTP-запроса
    
    # Проверяется, что текстовый запрос от пользователя не пустой
    if search_query.query: 
        
        # Формируется шаблон диалога: системная инструкция RAG + текущий вопрос человека
        prompt = ChatPromptTemplate.from_messages( 
            [
                ("system", template), # Системный промпт (правила поведения, контекст документов)
                ("human", "{input}"), # Переменная, куда подставится вопрос пользователя
            ]
        ) 
        
        # База Chroma превращается в поисковик с ограничением: извлекать ровно 3 самых релевантных документа
        retriever = chroma.as_retriever(search_kwargs={"k": 3}) 
        
        # Создается цепочка, которая берет документы и "втискивает" (stuff) их в контекст промпта для LLM
        combine_docs_chain = create_stuff_documents_chain(llm, prompt) 
        
        # Создается финальная RAG-цепочка: сначала ищет документы через retriever, затем передает их в combine_docs_chain
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain) 
        
        # Асинхронно запускается RAG-цепочка с передачей пользовательского вопроса в переменную "input"
        output = await retrieval_chain.ainvoke({"input": search_query.query}) 
        
        # Возвращается итоговый словарь с ответом модели и списком использованных документов (источников)
        return output 
