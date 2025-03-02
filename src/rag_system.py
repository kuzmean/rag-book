from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferWindowMemory  # new import

class RAGSystem:
    def __init__(self, model_name: str):
        self.llm = ChatOpenAI(temperature=0, model_name=model_name, base_url="https://api.proxyapi.ru/openai/v1")
        self.embeddings = OpenAIEmbeddings(base_url="https://api.proxyapi.ru/openai/v1")
        self.memory = ConversationBufferWindowMemory(k=2)  # optimized memory from LangChain
        
    def initialize_from_docs(self, documents):
        self.db = documents  # теперь documents это уже готовое векторное хранилище
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})  # Увеличиваем до 3 документов
        
        # Определяем функцию для форматирования документов с информацией об источниках
        def format_docs(docs):
            formatted_docs = []
            for doc in docs:
                chapter = doc.metadata.get('chapter', 'Неизвестная глава')
                page = doc.metadata.get('page', 'Неизвестная страница')
                # Добавляем метаданные в начало каждого документа
                formatted_text = f"[{chapter}, {page}]\n\n{doc.page_content}\n\n---\n"
                formatted_docs.append(formatted_text)
            return "\n".join(formatted_docs)
        
        prompt = ChatPromptTemplate.from_template('''
            Answer the user's question using only the provided context. 
            If the context does not contain enough information to answer the question, say: 
            "I cannot answer that question because the provided context does not contain relevant information."
            In this case, do NOT include any chapter or page information.
            Do not use any internal knowledge or information outside the provided context.

            ALWAYS include at least one direct quote from the text in your answer. Format quotes like this:
            «цитата из текста»
            
            After your complete answer, ON A NEW LINE, list all sources you used in your answer:
            
            Источники:
            [Глава X, Страницы Y, Z, W]
            
            Include ONLY sources that you actually quoted or referenced in your answer.
            Group sources by chapter - if you cited multiple pages from the same chapter, list them in a single line.
            For example, instead of:
            [Глава 6. Аристотель и Ликей, 160]
            [Глава 6. Аристотель и Ликей, 158]
            [Глава 6. Аристотель и Ликей, 161]
            
            Write:
            [Глава 6. Аристотель и Ликей, 158, 160, 161]
            
            List pages in ascending order. List each chapter on a separate line.
            
            Always respond in Russian.
            Conversation History: {history}
            Context: {context}
            Question: {input}
            Source Documents: {source_documents}
            Answer:
        ''')
        
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        # Используем базовую версию без дополнительных параметров
        self.retrieval_chain = create_retrieval_chain(
            self.retriever, 
            document_chain
        )
    
    def get_answer(self, question: str) -> str:
        # Load conversation history from optimized memory
        history = self.memory.load_memory_variables({}).get("history", "")
        
        # Получаем документы напрямую
        docs = self.retriever.get_relevant_documents(question)
        
        # Форматируем документы вручную
        formatted_docs = []
        for doc in docs:
            chapter = doc.metadata.get('chapter')
            page = doc.metadata.get('page')
            source_id = f"[{chapter}, {page}]"
            formatted_text = f"ИСТОЧНИК: Глава: {chapter} | Страница: {page}\n\n{doc.page_content}\n\n---\n"
            formatted_docs.append(formatted_text)
        
        formatted_context = "\n".join(formatted_docs)
        
        # Вызываем цепочку с отформатированным контекстом
        response = self.retrieval_chain.invoke({
            'input': question, 
            'history': history,
            'context': formatted_context,  # Передаем отформатированный контекст напрямую
            'source_documents': formatted_docs  # Добавляем исходные документы
        })
        
        answer = response.get('answer', 'Не удалось получить ответ от системы.')
        
        # Update memory with the interaction
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)
        return answer