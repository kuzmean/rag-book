from config import MODEL_NAME
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from bot import RAGBot

def main():
    # Инициализация RAG системы
    pdf_file = 'docs/История философии.pdf'
    rag_system = RAGSystem(MODEL_NAME)
    vectorstore = DocumentProcessor.process_pdf(pdf_file, rag_system.embeddings)
    
    # Создание и инициализация RAG системы
    rag_system.initialize_from_docs(vectorstore)
    
    # Запуск бота
    bot = RAGBot(rag_system)
    bot.start()

if __name__ == "__main__":
    main() 