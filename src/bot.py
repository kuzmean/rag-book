import telebot
from config import TELEGRAM_TOKEN

class RAGBot:
    def __init__(self, rag_system):
        self.bot = telebot.TeleBot(TELEGRAM_TOKEN)
        self.rag_system = rag_system
        self._setup_handlers()
        
    def _setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def send_welcome(message):
            welcome_text = """👋 Здравствуйте! Я бот-ассистент по книге "История философии".
            
            🤖 Я могу ответить на ваши вопросы о содержании книги, используя технологию RAG (Retrieval-Augmented Generation).
            
            📚 Просто задайте мне вопрос о книге, и я постараюсь найти релевантную информацию и предоставить вам точный ответ.
            
            ❗️ Важно: Я отвечаю только на основе содержания книги по античной философии и не использую внешние источники информации."""
            self.bot.reply_to(message, welcome_text)
            
        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            try:
                answer = self.rag_system.get_answer(message.text)
                self.bot.reply_to(message, answer)
            except Exception as e:
                error_message = f"Произошла ошибка: {str(e)}"
                self.bot.reply_to(message, error_message)
                
        @self.bot.message_handler(content_types=['audio', 'video', 'document', 'photo', 'sticker', 'voice', 'location', 'contact'])
        def not_text(message):
            self.bot.send_message(message.chat.id, 'Я работаю только с текстовыми сообщениями!')
    
    def start(self):
        print("Бот запущен...")
        self.bot.polling(none_stop=True) 