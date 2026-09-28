"""
core_filters.py

Консолидированная логика фильтрации/скоринга кандидатов, полученная в результате EDA
над train.parquet / benchmark_queries.parquet / benchmark_items.parquet.

- Жёсткий (отсекающий) фильтр стоит применять ТОЛЬКО к тому, что провалидировано на
  реальных парах (запрос -> реально выбранное объявление) из train с надёжностью ~98%+:
  категория (search_category / item_category_id, с особым случаем search_category == 0),
  порог рейтинга и метка "Вид услуги" из infm_params_text.
- Всё остальное, извлекаемое из infm_params_text (Тип услуги, Тип услуги автосервиса,
  Кто оказывает услуги и т.д.), показало заметно более низкую надёжность (от 9% до 87%
  на реальных данных) - отсекать по ним нельзя, это ощутимо режет Recall@50. Такие
  фильтры стоит использовать только как ДОПОЛНИТЕЛЬНЫЙ сигнал для ранжирования (буст
  скора), а не как условие отбора.
- Локация (search_location_id / item_location_id) — НЕ фильтр вообще, ни точный, ни по
  радиусу: несовпадения могут быть на тысячи км, надёжного радиуса не нашлось. Тоже
  soft-сигнал максимум.

Использование:
    from core_filters import (
        to_float, load_labels, build_label_pattern, parse_infm_params,
        extract_rating_threshold, category_ok, rating_ok, hard_filter_ok, soft_match_score,
    )
"""

import re
from collections import defaultdict

import pandas as pd

# ----------------------------------------------------------------------------------
# Список известных названий фильтров (из ручной курации по частотному анализу
# infm_params_text). Если рядом лежит файл nazvaniya_filtrov.csv -- используется он;
# иначе используется этот встроенный список как запасной вариант.
# ----------------------------------------------------------------------------------
DEFAULT_LABELS = [
    "Услуга", "Услуги", "Название услуги", "Тип услуги", "Вид услуги",
    "Тип услуги автосервиса", "Слова в описании", "Рейтинг пользователя 4 звезды и выше",
    "Чем вы занимаетесь", "Предмет или специальность", "Специальность",
    "Специальность или сфера", "Вид товара", "Тип товара", "Направление",
    "Кто оказывает услуги", "Чем занимается исполнитель", "Ваши клиенты",
    "Где вы оказываете услуги", "Место оказания услуг", "Место сделки", "Место работы",
    "Место проживания", "Место осмотра", "Учебное учреждение", "Название компании",
    "Название заведения", "Должность", "Сфера деятельности", "Язык перевода",
    "Иностранные языки", "Тип стоимости", "Начальная цена", "Предоплата", "Гарантия",
    "Опыт работы", "Грузоподъёмность", "Модель", "Марка", "Марка авто", "Тип кузова",
    "Тип техники", "Транспорт", "Цвет", "Жанр или формат", "Формат", "График работы",
    "Время работы", "Время для связи", "Рабочие дни", "Дни недели",
    "График работы, дни недели", "Продолжительность", "Возраст", "Пол", "Рейтинг",
    "Год окончания", "Площадь", "Длина", "Высота", "Какими комнатами занимаетесь",
    "Что перевозите", "Онлайн-запись",
]

# Метки, провалидированные на train как безопасные для ЖЁСТКОГО (отсекающего) фильтра.
# "Вид услуги": 98.4% пройденных реальных пар (N=316 860) -- безопасно.
# Всё остальное показало заметно меньшую надёжность -- см. docstring выше.
HARD_LABELS = {"Вид услуги"}

# Метки, которые нельзя сравнивать как обычный список значений -- у них другая природа.
PRESENCE_ONLY_LABELS = {"Онлайн-запись"}   # артефакт виджета бронирования: важно только наличие
DESCRIPTION_LABELS = {"Слова в описании"}   # сравнивать нужно с item_description_raw, не с infm_params

RATING_RE = re.compile(r"Рейтинг\s+пользователя\s+([\d]+(?:[.,]\d+)?)\s*звезд\w*\s*и\s*выше",
                        re.IGNORECASE)


# ----------------------------------------------------------------------------------
# Числовые поля (item_price / item_latitude / item_longitude приходят из parquet как
# object/Decimal, а не float -- нужна явная конвертация)
# ----------------------------------------------------------------------------------
# Приводит колонку к float. item_price / item_latitude / item_longitude приходят
# из parquet как object (Decimal), поэтому нужна явная конвертация.
# Вход: pd.Series. Выход: pd.Series[float]; нечисловые значения -> NaN.
def to_float(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


# ----------------------------------------------------------------------------------
# Категория: search_category == 0 значит "категория не выбрана" (в train таких строк
# исчезающе мало -- 34 из 497 673 -- а в benchmark_queries уже 9.05%; item_category_id
# никогда не равен 0). Если 0 -- не фильтруем вообще. Иначе -- жёсткое совпадение
# (99.99% на реальных train-парах).
# ----------------------------------------------------------------------------------
# Жёсткая проверка категории.
# search_category == 0 трактуем как "категория не выбрана" -> фильтр не применяем (True).
# Иначе объявление проходит, только если категории совпадают.
# Вход: search_category, item_category_id (числа). Выход: bool.
def category_ok(search_category, item_category_id):
    if search_category == 0:
        return True
    return search_category == item_category_id


# ----------------------------------------------------------------------------------
# Рейтинг: порог -- жёсткий фильтр (99%+ надёжность на train). Отсутствие рейтинга
# (NaN) считаем ПРОХОДЯЩИМ фильтр -- безопаснее для recall, чем отбрасывать.
# ----------------------------------------------------------------------------------
# Достаёт числовой порог из фильтра "Рейтинг пользователя N звезды и выше".
# Вход: текст (search_infm_params_text либо сама метка). Выход: float или None, если порога нет.
def extract_rating_threshold(text):
    if not isinstance(text, str) or not text:
        return None
    m = RATING_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


# Проверка порога рейтинга у объявления.
# Нет порога или у объявления нет рейтинга (NaN/None) -> считаем, что фильтр пройден
# (так безопаснее для recall). Иначе item_rating >= threshold. Выход: bool.
def rating_ok(item_rating, threshold):
    if threshold is None:
        return True
    if item_rating is None or pd.isna(item_rating):
        return True
    return float(item_rating) >= threshold


# ----------------------------------------------------------------------------------
# Разбор infm_params_text по известным меткам.
# ----------------------------------------------------------------------------------
# Загружает список известных названий фильтров: из CSV (первая колонка), а если файла нет
# или он пуст -- из встроенного DEFAULT_LABELS. Метки сортируются по убыванию длины,
# чтобы более длинная метка ("Тип услуги автосервиса") матчилась раньше короткой ("Тип услуги").
# Выход: list[str].
def load_labels(path=None):
    if path:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            col = df.columns[0]
            labels = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
            if labels:
                return sorted(set(labels), key=len, reverse=True)
        except FileNotFoundError:
            pass
    return sorted(set(DEFAULT_LABELS), key=len, reverse=True)


# Собирает регулярное выражение-альтернацию по всем меткам фильтров.
# Границы слова защищают от ложных срабатываний на коротких метках ("Пол" в "Полины").
# Вход: list[str] меток. Выход: скомпилированный re.Pattern (с группой захвата -- нужна для re.split).
def build_label_pattern(labels):
    # (?<![а-яёА-ЯЁa-zA-Z]) -- метка не должна начинаться внутри другого слова
    #                          (иначе "Пол" матчится в "Полины", "Цвет" -- в "Цветочная")
    # (?![а-яё])            -- после метки не должна сразу идти строчная буква
    #                          (иначе это продолжение того же слова, а не конец метки)
    body = "|".join(re.escape(l) for l in labels)
    return re.compile(r"(?<![а-яёА-ЯЁa-zA-Z])(" + body + r")(?![а-яё])")


# Нормализация строки для сравнения: схлопывает пробелы, обрезает края, приводит к нижнему регистру.
def normalize(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


# Разбирает склеенный текст infm_params_text на словарь {метка: [значения]}.
# Возвращает (словарь, leftover), где leftover -- текст до первой метки.
# Метка попадает в словарь даже с пустым значением (важно для флагов вроде рейтинга).
# Несколько значений одной метки = multi-select (внутри метки правило OR).
def parse_infm_params(text, pattern):
    """Возвращает (dict: метка -> список значений, leftover-текст до первой метки).

    ВАЖНО: метка попадает в результат, даже если у неё пустое значение (например, когда
    сразу за ней идёт следующая известная метка без текста между ними) -- иначе такие
    метки-флаги (в т.ч. "Рейтинг пользователя N звезды и выше", когда она стоит прямо
    перед другой меткой) молча теряются и проверки по ним не срабатывают.
    """
    if not isinstance(text, str) or not text.strip():
        return {}, ""
    parts = pattern.split(text)
    leftover = parts[0].strip()
    result = defaultdict(list)
    i = 1
    while i < len(parts) - 1:
        label = parts[i]
        value = parts[i + 1].strip()
        _ = result[label]  # гарантирует ключ в результате, даже если value пустой
        if value:
            result[label].append(value)
        i += 2
    return dict(result), leftover


# ----------------------------------------------------------------------------------
# Итоговая логика: жёсткий фильтр (используется чтобы ОТСЕЯТЬ кандидата) +
# мягкий скор (используется чтобы РАНЖИРОВАТЬ прошедших фильтр кандидатов).
# ----------------------------------------------------------------------------------
# Построчная (эталонная) версия ЖЁСТКОГО фильтра: категория + порог рейтинга + метка "Вид услуги".
# Остальные метки здесь намеренно не проверяются -- они идут в мягкий скор (soft_match_score).
# В боевом пайплайне (candidates_pipeline.py) та же логика реализована быстрее -- векторными масками
# по всему корпусу. Вход: параметры запроса и объявления + скомпилированный pattern. Выход: bool.
def hard_filter_ok(search_category, item_category_id, search_text, item_text, item_rating, pattern):
    """True/False -- проходит ли объявление жёсткий (отсекающий) фильтр.

    Включает: категорию, порог рейтинга (если есть в search_text), метку "Вид услуги".
    Всё остальное намеренно НЕ проверяется здесь -- см. soft_match_score.
    """
    if not category_ok(search_category, item_category_id):
        return False

    search_filters, _ = parse_infm_params(search_text, pattern)

    for label in search_filters:
        if label.startswith("Рейтинг пользователя"):
            threshold = extract_rating_threshold(label)
            if not rating_ok(item_rating, threshold):
                return False

    if "Вид услуги" in search_filters:
        item_filters, _ = parse_infm_params(item_text, pattern)
        search_values = {normalize(v) for v in search_filters["Вид услуги"]}
        item_values = item_filters.get("Вид услуги")
        if item_values is None:
            # метки нет у объявления вовсе -- мягкий фолбэк на сырой текст
            item_text_norm = normalize(item_text) if isinstance(item_text, str) else ""
            if not any(v in item_text_norm for v in search_values):
                return False
        else:
            item_set = {normalize(v) for v in item_values}
            if not (search_values & item_set):
                return False

    return True


# Мягкий скор: доля НЕжёстких фильтров запроса, совпавших с объявлением (0..1).
# Правила сравнения: presence-метки -- проверка наличия метки; "Слова в описании" -- поиск в описании;
# если метки нет у объявления -- поиск значения в сыром тексте; иначе пересечение значений (OR).
# Если фильтров нет, возвращает 1.0. В пайплайне используется быстрый аналог soft_match_fast.
def soft_match_score(search_text, item_text, item_description, pattern):
    """Доля НЕ жёстких фильтров запроса, которые совпали с объявлением (0..1) --
    использовать как буст к скору ранжирования (BM25/эмбеддинги), а не как отсекающий
    фильтр -- на реальных данных надёжность этих меток от ~9% до ~87%, отсекать по ним
    нельзя без заметной потери Recall@50.
    """
    search_filters, _ = parse_infm_params(search_text, pattern)
    if not search_filters:
        return 1.0  # фильтров нет -- нечего проверять, не наказываем и не поощряем

    item_filters, _ = parse_infm_params(item_text, pattern)
    item_text_norm = normalize(item_text) if isinstance(item_text, str) else ""
    item_desc_norm = normalize(item_description) if isinstance(item_description, str) else ""

    total = 0
    matched = 0
    for label, values in search_filters.items():
        if label in HARD_LABELS or label.startswith("Рейтинг пользователя"):
            continue  # уже учтено в hard_filter_ok

        total += 1
        search_values = {normalize(v) for v in values}

        if label in PRESENCE_ONLY_LABELS:
            if label in item_filters:
                matched += 1
            continue

        if label in DESCRIPTION_LABELS:
            if any(v in item_desc_norm for v in search_values):
                matched += 1
            continue

        item_values = item_filters.get(label)
        if item_values is None:
            if any(v in item_text_norm for v in search_values):
                matched += 1
            continue

        item_set = {normalize(v) for v in item_values}
        if search_values & item_set:
            matched += 1

    return matched / total if total else 1.0


# ----------------------------------------------------------------------------------
# Пример использования (не выполняется при импорте)
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    labels = load_labels("nazvaniya_filtrov.csv")  # или load_labels() для встроенного списка
    pattern = build_label_pattern(labels)

    search_text = "Рейтинг пользователя 4 звезды и выше Вид услуги Красота, здоровье"
    item_text = "Вид услуги Красота, здоровье Тип услуги СПА-услуги, массаж Место оказания услуг Москва"

    ok = hard_filter_ok(
        search_category=114, item_category_id=114,
        search_text=search_text, item_text=item_text, item_rating=4.7,
        pattern=pattern,
    )
    score = soft_match_score(search_text, item_text, item_description="", pattern=pattern)
    print("hard_filter_ok:", ok)
    print("soft_match_score:", score)
