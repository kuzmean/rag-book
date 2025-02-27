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
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        prompt = ChatPromptTemplate.from_template('''
            Answer the user's question using only the provided context. 
            If the context does not contain enough information to answer the question, say: 
            "I cannot answer that question because the provided context does not contain relevant information."
            Do not use any internal knowledge or information outside the provided context.
            Always respond in Russian.
            Conversation History: {history}
            Context: {context}
            Question: {input}
            Answer:
        ''')
        
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        self.retrieval_chain = create_retrieval_chain(
            self.retriever, 
            document_chain
        )
    
    def get_answer(self, question: str) -> str:
        # Load conversation history from optimized memory
        history = self.memory.load_memory_variables({}).get("history", "")
        response = self.retrieval_chain.invoke({'input': question, 'history': history})
        answer = response.get('answer', 'Не удалось получить ответ от системы.')
        # Update memory with the interaction
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)
        return answer