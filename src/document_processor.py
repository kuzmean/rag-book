from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os
from books import find_chapter_for_page

class DocumentProcessor:
    @staticmethod
    def process_pdf(file_path: str, embeddings):
        # Создаем имя для файла эмбеддингов на основе имени PDF
        pdf_name = os.path.basename(file_path)
        embeddings_dir = "embeddings"
        embeddings_path = os.path.join(embeddings_dir, os.path.splitext(pdf_name)[0])

        # Проверяем, существуют ли уже эмбеддинги
        if os.path.exists(embeddings_path):
            print("Загружаем существующие эмбеддинги...")
            return FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)

        print("Создаем новые эмбеддинги...")

        # Загружаем PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Очистка и нормализация текста
        for doc in documents:
            cleaned_content = doc.page_content.replace('\n', ' ').strip()
            doc.page_content = " ".join(cleaned_content.split())
            
            # Получаем номер страницы из метаданных
            page_num = doc.metadata.get("page", 0)
            
            # Добавляем информацию о главе из предопределенных метаданных
            chapter_title = find_chapter_for_page(pdf_name, page_num)
            if chapter_title:
                doc.metadata["chapter"] = chapter_title
                # Форматируем номер страницы
                doc.metadata["page"] = f"Страница {page_num}"
            
            # Фильтруем метаданные
            doc.metadata = {
                key: value 
                for key, value in doc.metadata.items() 
                if key in ["page", "chapter"]
            }
            
        # Разделяем документы на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(documents)
        
        # Создаем и сохраняем векторное хранилище
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Создаем директорию, если её нет
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Сохраняем эмбеддинги
        vectorstore.save_local(embeddings_path)
        
        return vectorstore