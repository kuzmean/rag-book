from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os

class DocumentProcessor:
    @staticmethod
    def process_pdf(file_path: str, embeddings):
        # Создаем имя для файла эмбеддингов на основе имени PDF
        pdf_name = os.path.splitext(os.path.basename(file_path))[0]
        embeddings_dir = "embeddings"
        embeddings_path = os.path.join(embeddings_dir, pdf_name)

        # Проверяем, существуют ли уже эмбеддинги
        if os.path.exists(embeddings_path):
            print("Загружаем существующие эмбеддинги...")
            return FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)

        print("Создаем новые эмбеддинги...")
        loader = UnstructuredPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = loader.load_and_split(text_splitter)
        
        # Добавляем метадату 'page' для каждого документа
        for i, doc in enumerate(documents):
            doc.metadata["page"] = f"Страница {i+1}"
        
        # Создаем и сохраняем векторное хранилище
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Создаем директорию, если её нет
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Сохраняем эмбеддинги
        vectorstore.save_local(embeddings_path)
        
        return vectorstore