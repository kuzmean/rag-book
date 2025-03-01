BOOK_METADATA = {
    "История философии.pdf": {
        "chapters": [
            {
                "title": "Введение",
                "start_page": 10,
                "end_page": 26
            },
            {
                "title": "Глава 1. Понятие досократической философии, ее место в истории античной философии",
                "start_page": 27,
                "end_page": 44
            },
            {
                "title": "Глава 2. Обзор основных философских позиций досократического периода",
                "start_page": 45,
                "end_page": 89
            },
            {
                "title": "Глава 3. Античная софистика",
                "start_page": 90,
                "end_page": 103
            },
            {
                "title": "Глава 4. Сократ и сократические школы",
                "start_page": 107,
                "end_page": 122
            },
            {
                "title": "Глава 5. Платон и Академия",
                "start_page": 124,
                "end_page": 153
            },
            {
                "title": "Глава 6. Аристотель и Ликей",
                "start_page": 154,
                "end_page": 196
            },
            {
                "title": "Глава 7. Стоицизм",
                "start_page": 201,
                "end_page": 213
            },
            {
                "title": "Глава 8. Эпикуреизм",
                "start_page": 214,
                "end_page": 223
            },
            {
                "title": "Глава 9. Скептицизм",
                "start_page": 225,
                "end_page": 231
            },
            {
                "title": "Глава 12. Неоплатонизм",
                "start_page": 253,
                "end_page": 270
            },
            {
                "title": "Глава 13. Влияние античной философии на средневековую мысль",
                "start_page": 272,
                "end_page": 286
            }
        ]
    }
}

def find_chapter_for_page(book_name, page_number):
    """Определяет главу по номеру страницы для указанной книги"""
    book_info = BOOK_METADATA.get(book_name, {})
    chapters = book_info.get("chapters", [])
    
    if chapters:
        for chapter in chapters:
            if chapter["start_page"] <= page_number <= chapter["end_page"]:
                return chapter["title"]
        return "Введение"  # По умолчанию
    return "Неизвестная глава"  # Если глав нет или книга не найдена 