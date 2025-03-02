import os
import sys
import json
import datetime
import random
import string
import time
from dotenv import load_dotenv
from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from openai import OpenAI

# Добавляем родительскую директорию в путь для импорта
# Добавляем src директорию (на одном уровне с validation) в путь для импорта
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_path)

# Импорт RAG системы
from config import MODEL_NAME
from document_processor import DocumentProcessor
from rag_system import RAGSystem

# Загрузка переменных окружения
load_dotenv()

# Упрощенная настройка клиента
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Проверка наличия ключей
if not os.environ["LANGSMITH_API_KEY"] or not os.environ["OPENAI_API_KEY"]:
    print("Ошибка: Отсутствуют необходимые API ключи")
    sys.exit(1)

# Инициализация клиентов
api_key = os.getenv("LANGSMITH_API_KEY", "")
client = Client(api_key=api_key)
openai_client = wrappers.wrap_openai(OpenAI(base_url="https://api.proxyapi.ru/openai/v1"))

def load_evaluation_results(file_path):
    """Загрузка результатов оценки"""
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        print(f"Загружено {len(results)} результатов из файла {file_path}")
        if len(results) < 20:
            print("ВНИМАНИЕ: Загружено менее 20 результатов! Возможно, набор данных неполный.")
        return results

def create_dataset_from_results(results, dataset_name="Философия RAG оценка"):
    """Создание датасета в LangSmith из результатов оценки"""
    # Добавляем уникальный суффикс к имени датасета
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choice(string.ascii_lowercase) for _ in range(4))
    unique_dataset_name = f"{dataset_name}_{timestamp}_{random_suffix}"
    
    print(f"Создание датасета '{unique_dataset_name}'...")
    print(f"Количество примеров для загрузки: {len(results)}")
    
    # Печать структуры первого примера для отладки
    if results:
        print("\nСтруктура первого примера:")
        print(f"Ключи: {results[0].keys()}")
        print(f"Query: {results[0].get('query', 'Отсутствует')}")
        print(f"Reference: {results[0].get('reference', 'Отсутствует')}")
    
    # Создание датасета
    try:
        dataset = client.create_dataset(
            dataset_name=unique_dataset_name,
            description="Датасет для оценки RAG системы по античной философии"
        )
        print(f"Датасет успешно создан с ID: {dataset.id}")
    except Exception as e:
        print(f"Ошибка при создании датасета: {e}")
        print("Попытка поиска существующих датасетов...")
        
        # Пытаемся найти существующие датасеты
        try:
            datasets = list(client.list_datasets())
            print("\nДоступные датасеты:")
            for i, ds in enumerate(datasets):
                print(f"{i+1}. {ds.name} (ID: {ds.id})")
            
            choice = input("\nВыберите номер датасета для использования (или Enter для выхода): ")
            if not choice:
                print("Операция отменена.")
                sys.exit(0)
                
            dataset = datasets[int(choice) - 1]
            print(f"Выбран датасет: {dataset.name} (ID: {dataset.id})")
            return dataset
        except Exception as search_error:
            print(f"Ошибка при поиске датасетов: {search_error}")
            sys.exit(1)
    
    # Подготовка входных и эталонных данных
    inputs = [{"question": item["query"]} for item in results]
    outputs = [{"reference_answer": item["reference"]} for item in results]
    
    print(f"Подготовлено {len(inputs)} входных примеров и {len(outputs)} эталонных ответов")
    
    # Добавление примеров в датасет
    try:
        client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
        print(f"Все {len(inputs)} примеров успешно добавлены в датасет.")
    except Exception as e:
        print(f"Ошибка при добавлении примеров: {e}")
        return dataset
    
    return dataset

def initialize_rag_system():
    """Инициализация RAG системы"""
    print("Инициализация RAG системы...")
    # Путь к PDF-файлу в корневой директории проекта
    pdf_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs/История философии.pdf')
    
    if not os.path.exists(pdf_file):
        print(f"Ошибка: Не удалось найти PDF файл по пути {pdf_file}")
        print("Убедитесь, что файл находится в директории docs/ в корне проекта")
        sys.exit(1)
    else:
        print(f"Найден PDF файл: {pdf_file}")
    
    rag_system = RAGSystem(MODEL_NAME)
    vectorstore = DocumentProcessor.process_pdf(pdf_file, rag_system.embeddings)
    rag_system.initialize_from_docs(vectorstore)
    return rag_system

# Целевая функция для оценки
def rag_target(inputs):
    """Функция для обработки запросов через RAG систему"""
    # Используем глобальную переменную для RAG системы
    global rag_instance
    
    # Проверяем, какой ключ доступен в входных данных
    if "question" in inputs:
        question = inputs["question"]
    elif "query" in inputs:
        question = inputs["query"]
    elif "input" in inputs:
        question = inputs["input"]
    else:
        # Если не найдено подходящих ключей, используем первый ключ
        question = next(iter(inputs.values()), "Без запроса")
        print(f"Предупреждение: Используется запрос без стандартного ключа: {question[:30]}...")
    
    answer = rag_instance.get_answer(question)
    return {"response": answer}

# Схема для оценки точности
class AccuracyGrade(BaseModel):
    score: float = Field(description="Оценка от 0 до 1, где 1 означает полное соответствие эталонному ответу")
    reasoning: str = Field(description="Объяснение оценки")

# Схема для оценки фактичности
class GroundednessGrade(BaseModel):
    score: float = Field(description="Оценка от 0 до 1, где 1 означает полностью фактический ответ")
    reasoning: str = Field(description="Объяснение оценки")

# Оценка точности ответа
def evaluate_accuracy(outputs, reference_outputs):
    """Оценка точности ответа по сравнению с эталонным"""
    instructions = """
    Оцените ответ системы по сравнению с эталонным ответом по шкале от 0 до 1:
    - 1.0: Полное соответствие всех ключевых концепций и фактов
    - 0.8: Большинство ключевых концепций правильны, есть незначительные неточности
    - 0.6: Частичное соответствие с некоторыми пропусками или ошибками
    - 0.4: Базовое понимание, но существенные пропуски
    - 0.2: Минимальное соответствие
    - 0.0: Полное несоответствие или неверная информация
    
    Оценивайте СОДЕРЖАНИЕ, а не точное соответствие слов. Важно совпадение ключевых концепций и фактов.
    """
    
    # Выбираем подходящий ключ для эталонного ответа
    reference_answer = None
    if "reference_answer" in reference_outputs:
        reference_answer = reference_outputs["reference_answer"]
    elif "reference" in reference_outputs:
        reference_answer = reference_outputs["reference"]
    elif "ground_truth" in reference_outputs:
        reference_answer = reference_outputs["ground_truth"]
    else:
        # Если подходящий ключ не найден, берем первое значение
        reference_answer = next(iter(reference_outputs.values()), "Эталонный ответ отсутствует")
        print(f"Предупреждение: Используется эталонный ответ без стандартного ключа")
    
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""
            Эталонный ответ: {reference_answer}
            
            Ответ системы: {outputs["response"]}
            """}
        ],
        response_format=AccuracyGrade
    )
    
    return response.choices[0].message.parsed.score

# Оценка фактичности ответа
def evaluate_groundedness(outputs, reference_outputs):
    """Оценка фактичности ответа (отсутствия галлюцинаций)"""
    instructions = """
    Оцените фактичность ответа системы по шкале от 0 до 1:
    - 1.0: Идеально фактический ответ, полностью основан на достоверной информации
    - 0.8: В основном фактический, с незначительными неточностями
    - 0.6: Частично фактический, есть некоторые необоснованные утверждения
    - 0.4: Много необоснованных утверждений, но есть и фактическая информация
    - 0.2: В основном содержит "галлюцинации", мало фактов
    - 0.0: Полностью выдуманная информация, не соответствующая действительности
    
    "Галлюцинации" - это утверждения, которые не основаны на фактах или искажают действительность.
    Оценивайте, насколько ответ соответствует историческим и философским фактам.
    """
    
    # Выбираем подходящий ключ для эталонного ответа
    reference_answer = None
    if "reference_answer" in reference_outputs:
        reference_answer = reference_outputs["reference_answer"]
    elif "reference" in reference_outputs:
        reference_answer = reference_outputs["reference"]
    elif "ground_truth" in reference_outputs:
        reference_answer = reference_outputs["ground_truth"]
    else:
        reference_answer = next(iter(reference_outputs.values()), "Эталонный ответ отсутствует")
    
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""
            Эталонный ответ: {reference_answer}
            
            Ответ системы для проверки на фактичность: {outputs["response"]}
            
            Оцените, насколько ответ системы основан на фактах, а не на "галлюцинациях".
            """}
        ],
        response_format=GroundednessGrade
    )
    
    return response.choices[0].message.parsed.score

# Схема для оценки полезности
class HelpfulnessGrade(BaseModel):
    score: float = Field(description="Оценка от 0 до 1, где 1 означает максимальную полезность")
    reasoning: str = Field(description="Объяснение оценки")

# Оценка полезности ответа
def evaluate_helpfulness(outputs, inputs):
    """Оценка полезности ответа для пользователя"""
    instructions = """
    Оцените полезность ответа системы для вопроса пользователя по шкале от 0 до 1:
    - 1.0: Исключительно полезный, полный и хорошо структурированный ответ
    - 0.8: Очень полезный ответ с хорошей структурой
    - 0.6: Достаточно полезный ответ, но могло быть лучше
    - 0.4: Минимально полезный ответ с существенными недостатками
    - 0.2: Очень мало полезной информации
    - 0.0: Бесполезный ответ
    
    Учитывайте формат, структуру, ясность и полноту ответа.
    """
    
    # Выбираем подходящий ключ для вопроса
    question = None
    if "question" in inputs:
        question = inputs["question"]
    elif "query" in inputs:
        question = inputs["query"]
    elif "input" in inputs:
        question = inputs["input"]
    else:
        question = next(iter(inputs.values()), "Вопрос отсутствует")
        print(f"Предупреждение: Используется вопрос без стандартного ключа")
    
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""
            Вопрос пользователя: {question}
            
            Ответ системы: {outputs["response"]}
            """}
        ],
        response_format=HelpfulnessGrade
    )
    
    return response.choices[0].message.parsed.score

def find_dataset_by_name(name_prefix):
    """Поиск датасета по части имени"""
    try:
        datasets = list(client.list_datasets())
        matching_datasets = [ds for ds in datasets if name_prefix.lower() in ds.name.lower()]
        
        if matching_datasets:
            print(f"\nНайдены датасеты, содержащие '{name_prefix}':")
            for i, ds in enumerate(matching_datasets):
                print(f"{i+1}. {ds.name} (ID: {ds.id})")
            
            choice = input("\nВыберите номер датасета для использования (или Enter для создания нового): ")
            if choice and choice.isdigit() and 0 < int(choice) <= len(matching_datasets):
                return matching_datasets[int(choice) - 1]
        
        return None
    except Exception as e:
        print(f"Ошибка при поиске датасетов: {e}")
        return None

def generate_test_dataset() -> List[Dict[str, Any]]:
    """Создает тестовый датасет с вопросами и эталонными ответами по древнегреческой философии"""
    print("Создание тестового датасета с вопросами по древнегреческой философии...")
    
    questions_and_answers = [
        {
            "query": "Кто такой Сократ?",
            "reference": "Сократ (469-399 до н.э.) - древнегреческий философ из Афин, один из основоположников западной философии. Он не оставил письменных трудов, а его учение известно благодаря сочинениям его учеников, особенно Платона. Сократ разработал метод диалектических споров (майевтику) и был известен своим высказыванием 'Я знаю, что ничего не знаю'."
        },
        {
            "query": "В чем суть теории идей Платона?",
            "reference": "Теория идей Платона утверждает, что истинная реальность состоит из вечных, неизменных идей (форм), а физический мир - лишь их тень или отражение. Каждый объект в материальном мире является несовершенной копией своей идеальной формы. Например, все конкретные круги в физическом мире - это несовершенные проявления идеи круга."
        },
        {
            "query": "Чем философия Аристотеля отличается от философии Платона?",
            "reference": "В отличие от Платона, Аристотель считал, что формы существуют в самих вещах, а не в отдельном мире идей. Аристотель был более ориентирован на эмпирические исследования и классификацию знаний. Он разработал логику, систему категорий и учение о четырех причинах, а также считал, что добродетель - это 'золотая середина' между крайностями."
        },
        {
            "query": "Кто основал стоицизм?",
            "reference": "Стоицизм был основан Зеноном Китийским около 300 г. до н.э. в Афинах. Своё название школа получила от портика (стоя), где Зенон обычно преподавал."
        },
        {
            "query": "Кто такие досократики?",
            "reference": "Досократики - это греческие философы, жившие до Сократа (примерно 6-5 вв. до н.э.). К ним относятся представители Милетской школы (Фалес, Анаксимандр, Анаксимен), пифагорейцы, Гераклит, Парменид, Зенон Элейский, Демокрит и другие. Они занимались преимущественно вопросами происхождения и устройства мира."
        },
        {
            "query": "Каковы основные идеи Эпикура?",
            "reference": "Эпикур учил, что высшее благо - это удовольствие, понимаемое как отсутствие страданий и душевного беспокойства. Он считал, что мир состоит из атомов и пустоты, боги не вмешиваются в человеческие дела, а смерть не следует бояться, так как 'когда мы есть, смерти нет, а когда смерть есть, нас уже нет'."
        },
        {
            "query": "Кто такой Диоген Синопский?",
            "reference": "Диоген Синопский (около 412-323 до н.э.) - древнегреческий философ, представитель школы киников. Он вел аскетичный образ жизни, отвергал социальные условности, жил в бочке. Согласно легенде, ходил днем с фонарем, 'ища человека'. Был известен своими эксцентричными выходками и остроумными ответами."
        },
        {
            "query": "Какое основное учение Гераклита?",
            "reference": "Основное учение Гераклита - о постоянном изменении и текучести бытия, выраженное фразой 'Всё течёт, всё меняется'. Он считал, что нельзя дважды войти в одну и ту же реку. Первоначалом всего Гераклит считал огонь. Он также ввел понятие логоса - универсального закона, управляющего космосом."
        },
        {
            "query": "Что такое атомизм Демокрита?",
            "reference": "Атомизм Демокрита - учение о том, что мир состоит из мельчайших, неделимых частиц (атомов) и пустоты. Атомы вечны, различаются формой и размером, их движение и соединение объясняют все изменения в мире. Демокрит считал, что даже душа состоит из особых, круглых и подвижных атомов."
        },
        {
            "query": "Кто такие софисты?",
            "reference": "Софисты - группа учителей красноречия и философов в Древней Греции V-IV вв. до н.э., включая Протагора и Горгия. Они первыми стали брать плату за обучение, обучали риторике и аргументации. Софисты отличались релятивизмом - учением об относительности истины и морали, что выражено в знаменитом высказывании Протагора: 'Человек есть мера всех вещей'."
        },
        {
            "query": "Что открыли пифагорейцы?",
            "reference": "Пифагорейцы открыли числовые соотношения в музыкальной гармонии, разработали теорему Пифагора, внесли вклад в развитие математики и геометрии. Они также предложили идею о сферической форме Земли и других небесных тел, ввели понятие космоса как упорядоченной Вселенной."
        },
        {
            "query": "В чем главная идея Парменида?",
            "reference": "Главная идея Парменида заключается в том, что существует только неизменное, вечное и единое бытие. Он утверждал, что 'бытие есть, а небытия нет', отрицал возможность возникновения и уничтожения. Парменид считал чувственный мир с его множественностью и изменениями иллюзией, а истинная реальность постижима только разумом."
        },
        {
            "query": "Кто основал Милетскую школу?",
            "reference": "Милетскую школу основал Фалес из Милета, которого считают первым греческим философом. Вместе с Анаксимандром и Анаксименом они создали первую натурфилософскую школу в Древней Греции в VI веке до н.э., стремившуюся найти материальное первоначало (архэ) всего сущего."
        },
        {
            "query": "Как Аристотель классифицировал науки?",
            "reference": "Аристотель классифицировал науки на три группы: теоретические (физика, математика, метафизика), практические (этика, политика) и творческие (поэтика, риторика). Теоретические науки изучают истину ради нее самой, практические направлены на действие и поведение, а творческие связаны с созданием произведений."
        },
        {
            "query": "Каковы три класса в идеальном государстве Платона?",
            "reference": "В идеальном государстве Платона общество разделено на три класса: 1) правители-философы, обладающие мудростью; 2) стражи-воины, отличающиеся мужеством; 3) производители (ремесленники, земледельцы), проявляющие умеренность. Каждый класс выполняет свою функцию, что обеспечивает справедливость в государстве."
        },
        {
            "query": "Что такое эвдемония у Аристотеля?",
            "reference": "Эвдемония у Аристотеля - это высшее благо, блаженство или счастье, достигаемое через добродетельную деятельность в соответствии с совершенством души. Это не просто удовольствие, а полная и совершенная жизнь, включающая как практику добродетелей, так и достаточное количество внешних благ."
        },
        {
            "query": "Для чего Зенон Элейский придумал свои парадоксы?",
            "reference": "Зенон Элейский придумал свои парадоксы (апории) для защиты учения Парменида о неизменном и едином бытии. Парадоксы показывали логические противоречия в понятиях множественности и движения, демонстрируя несостоятельность представлений о мире, основанных на чувственном опыте."
        },
        {
            "query": "Что такое апейрон Анаксимандра?",
            "reference": "Апейрон Анаксимандра - это 'беспредельное', неопределенное первоначало всего сущего. В отличие от Фалеса, который считал первоначалом воду, Анаксимандр предложил идею неопределенной, бесконечной субстанции, из которой возникают все вещи и в которую они возвращаются при разрушении."
        },
        {
            "query": "Какой метод исследования использовал Сократ?",
            "reference": "Сократ использовал диалектический метод исследования, известный как майевтика ('повивальное искусство') — метод вопросов и ответов, помогающий собеседнику самому прийти к истине. Сократ задавал вопросы, выявлял противоречия в ответах, помогая 'родиться' новому знанию, подобно тому как повитуха помогает родиться ребенку."
        },
        {
            "query": "Какое значение имело число в философии пифагорейцев?",
            "reference": "В философии пифагорейцев число имело фундаментальное значение. Они считали, что числа являются сущностью всех вещей ('всё есть число'). Пифагорейцы обнаружили, что музыкальные интервалы могут быть выражены в числовых пропорциях, и распространили эту идею на космос, веря, что вся Вселенная устроена согласно числовой гармонии."
        },
        {
            "query": "Что Аристотель называл энтелехией?",
            "reference": "Энтелехия в философии Аристотеля - это актуальное осуществление потенциальной возможности формы, внутренняя цель движения. Это принцип, который превращает возможность в действительность. Например, желудь содержит в себе энтелехию дуба - внутреннюю цель стать взрослым деревом."
        }
    ]
    
    # Создаем полный датасет с метаданными
    full_dataset = []
    for qa in questions_and_answers:
        # Симулируем ответ RAG системы (в реальности это должно быть от вашей RAG)
        # Но так как мы просто создаем тестовый датасет, используем заглушку
        response = "Ответ будет сгенерирован RAG системой при запуске эксперимента."
        
        entry = {
            "query": qa["query"],
            "reference": qa["reference"],
            "response": response,
            "response_time": 0.0,  # Заглушка для времени ответа
            "metadata": {"source": "test_dataset", "created": time.time()}
        }
        full_dataset.append(entry)
    
    print(f"Создан тестовый датасет с {len(full_dataset)} вопросами по древнегреческой философии")
    return full_dataset

def save_dataset(dataset, file_path):
    """Сохраняет датасет в JSON файл"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Датасет сохранен в {file_path}")

def main():
    # Загрузка результатов
    # Используем путь относительно текущего файла
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    
    # Проверяем наличие файла результатов, если нет - создаем тестовый датасет
    created_new_dataset = False
    if os.path.exists(results_path):
        print(f"Загрузка существующих результатов из {results_path}...")
        results = load_evaluation_results(results_path)
    else:
        print(f"Файл {results_path} не найден. Создаем тестовый датасет...")
        results = generate_test_dataset()
        created_new_dataset = True
        
        # Спрашиваем пользователя, хочет ли он сохранить тестовый датасет
        save_option = input("Сохранить тестовый датасет для будущего использования? (y/n): ").lower()
        if save_option == 'y':
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            save_dataset(results, results_path)
    
    # Создание датасета
    print("Подготовка данных для эксперимента...")
    experiment_name = input("Введите название эксперимента (по умолчанию: philosophy-rag-evaluation): ")
    if not experiment_name:
        experiment_name = "philosophy-rag-evaluation"
    
    # Инициализация RAG системы для оценки
    global rag_instance
    rag_instance = initialize_rag_system()
    
    # Всегда создаем новый датасет в LangSmith из наших данных
    print(f"Загрузка {len(results)} примеров в LangSmith для тестирования...")
    dataset = create_dataset_from_results(results, f"Древнегреческая философия - {experiment_name}")
    dataset_id = dataset.id
    print(f"Датасет успешно загружен в LangSmith с ID: {dataset_id}")
    
    print("\nЗапуск эксперимента в LangSmith...")
    experiment_results = client.evaluate(
        rag_target,
        data=dataset_id,
        evaluators=[
            evaluate_accuracy,
            evaluate_helpfulness,
            evaluate_groundedness
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2
    )
    
    print("\nЭксперимент успешно запущен!")
    print(f"Результаты доступны по ссылке: https://smith.langchain.com/projects")
    
    # Вывод метрик
    # Эксперимент запускается асинхронно в LangSmith
    print("\nОбработка результатов:")
    print("Эксперимент выполняется в LangSmith асинхронно. Результаты будут доступны в веб-интерфейсе.")
    print(f"Тип возвращаемых данных: {type(experiment_results)}")
    
    # Получаем ID эксперимента для создания прямой ссылки
    experiment_id = None
    try:
        if hasattr(experiment_results, 'id'):
            experiment_id = experiment_results.id
        elif hasattr(experiment_results, 'experiment_id'):
            experiment_id = experiment_results.experiment_id
        
        if experiment_id:
            print(f"\nПрямая ссылка на эксперимент: https://smith.langchain.com/experiments/{experiment_id}")
        
        # Для получения метрик после завершения эксперимента можно использовать:
        print("\nДля просмотра результатов после завершения эксперимента:")
        print("1. Перейдите по ссылке выше")
        print("2. Дождитесь завершения всех тестов")
        print("3. Просмотрите метрики в разделе 'Evaluations'")
    except Exception as e:
        print(f"Ошибка при получении ID эксперимента: {e}")
        print("Проверьте результаты в веб-интерфейсе LangSmith")

if __name__ == "__main__":
    main() 