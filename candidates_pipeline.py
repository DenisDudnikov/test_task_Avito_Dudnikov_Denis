"""
candidates_pipeline.py

Пайплайн генерации кандидатов для Recall@50.

Источники кандидатов и их приоритет:
  0. lookup-слой: точное совпадение текста запроса с train, если у него есть
     item_id, присутствующий в benchmark_items (высокий приоритет, но не
     обходит категорийный фильтр -- отсекать по нему рискованно, а лишний
     кандидат ничего не стоит).
  1. BM25 по item_title_raw (x3 вес) + item_infm_params_text + item_description_raw,
     после жёсткого AND-фильтра: категория + рейтинг + "Вид услуги".
  2. Если кандидатов < 50 -- тиры-фолбэки, постепенно ослабляющие жёсткий
     фильтр (сначала снимаем "Вид услуги", потом рейтинг, потом категорию) -
     фильтры это надёжные, но не 100%-но надёжные сигналы, поэтому вместо
     жёсткого обрыва используем ступенчатое ослабление, чтобы не терять recall
     на пограничных случаях.
  3. Если и это не даёт 50 -- добор самыми популярными объявлениями той же
     категории (по количеству отзывов).

Локация: НЕ жёсткий фильтр (несовпадения бывают на тысячи км), но:
  - плавный бонус к скору по расстоянию (exp(-dist/PROXIMITY_DECAY_KM), полный
    LOCATION_BONUS при точном совпадении location_id, гладко убывает дальше);
  - плюс адаптивное СУЖЕНИЕ: если после текста+фильтра кандидатов много (частый
    шаблонный запрос вроде "маникюр" - таких много в любом городе) - сначала
    пробуем ограничиться той же локацией, если мало - расширяем радиус, пока не
    наберём разумный пул (см. LOCATION_NARROW_* ниже). Это ДОПОЛНИТЕЛЬНЫЙ
    приоритетный тир поверх старых - если сужение не даёт достаточно кандидатов,
    старые (без учёта локации) тиры остаются как страховка recall.
  - при search_is_delivery_search==1 локация не учитывается вовсе (ни бонусом,
    ни сужением) - считаем, что для таких запросов расстояние не важно.

Запуск:
    python candidates_pipeline.py \
        --train train.parquet \
        --queries benchmark_queries.parquet \
        --items benchmark_items.parquet \
        --labels nazvaniya_filtrov.csv \
        --output answer.csv

Для подбора порогов (LOCATION_NARROW_*, PROXIMITY_DECAY_KM и т.д.) на train -
см. validate_holdout.py рядом.
"""

import argparse
import gc
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from core_filters import (
    HARD_LABELS,
    PRESENCE_ONLY_LABELS,
    DESCRIPTION_LABELS,
    build_label_pattern,
    extract_rating_threshold,
    load_labels,
    normalize,
    parse_infm_params,
    to_float,
)

# ----------------------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------------------
TOP_K = 50
POOL_SIZE = 400
BM25_K1 = 0.9
BM25_B = 0.5
TITLE_REPEAT = 3
MIN_DF = 2
MAX_FEATURES = 400_000

LOCATION_BONUS = 0.338     # бонус к BM25-скору за совпадение локации (не для delivery-запросов)
SOFT_MATCH_WEIGHT = 2.979  # вес soft_match_score при ре-ранжировании пула кандидатов

# --- Адаптивное сужение по локации --------------------------------------------------
# Идея: чем чаще/типовее запрос ("маникюр"), тем больше
# в корпусе текстово-похожих кандидатов и тем СИЛЬНЕЕ реальный ответ локален (наблюдали
# 93.6% совпадения локации у уникальных запросов против 97-98% у частых, при этом медиана
# расстояния падает с 5.4 км до ~4 км). Одновременно жёстко резать по локации нельзя -
# у части категорий (например, "Обучение, курсы") хвост расстояний уходит за тысячи км.
# Поэтому: если кандидатов после текстового поиска + жёсткого фильтра МНОГО - сужаем по
# локации (сначала тот же location_id, потом расширяем радиус), но это ДОПОЛНИТЕЛЬНЫЙ
# приоритетный тир, а не замена старых - если сужение не наберёт достаточно кандидатов,
# существующие тиры (без учёта локации) остаются как страховка для recall.
LOCATION_NARROW_TRIGGER = 300    # если кандидатов (hard-filter & text_score>0) больше - сужаем
LOCATION_NARROW_MIN_POOL = 20    # целевой минимум кандидатов после сужения
LOCATION_RADIUS_STEPS_KM = [20, 60, 150, 300, 800]  # шаги расширения радиуса до набора MIN_POOL
PROXIMITY_DECAY_KM = 29.88        # характерный масштаб плавного бонуса за близость (см. ниже)
EXTEND_LOCATIONS = True          # достраивать координаты для search_location_id, которых нет среди объявлений
MAX_LOCATION_SPREAD_KM = 100.0   # верхняя граница масштаба бонуса для таких "укрупнённых" локаций
EXTEND_MAX_SPREAD_KM = 50.0      # локации с большим разбросом остаются "неизвестными"

STOPWORDS = frozenset("""
а без более был была были было быть в вам вас весь во вот все всего всех вы
да для до его ее ей ему если есть ещё же за из или им их к как ко когда
кто ли либо мне мы на надо наш не него нее нет ни них но ну о об однако он
она они оно от очень по при с со та так также такой там те тем то того
тоже той только том ты у уже хотя чего чей чем что чтобы эта эти это я
""".split())

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

try:
    from nltk.stem.snowball import SnowballStemmer

    _stemmer = SnowballStemmer("russian")
    STEMMING_ENABLED = True
    _stem_cache = {}

    # Стемминг одного токена Snowball-стеммером с мемоизацией (nltk доступен).
    def _stem(tok):
        # Мемоизация критична: одни и те же слова встречаются в текстах объявлений
        # десятки тысяч раз, а SnowballStemmer.stem() на каждый вызов заметно
        # дороже простого dict-lookup - без кэша индексация корпуса из ~190k
        # объявлений становится главным узким местом пайплайна.
        cached = _stem_cache.get(tok)
        if cached is None:
            cached = _stemmer.stem(tok)
            _stem_cache[tok] = cached
        return cached
except Exception:
    STEMMING_ENABLED = False
    def _stem(tok):
        return tok


# Токенизатор для BM25: нижний регистр, слова из букв/цифр, длина > 1, без стоп-слов, со стеммингом.
def tokenize(text):
    """Токенизатор для BM25: слова из кириллицы/латиницы/цифр, без стоп-слов,
    с опциональным стеммингом (nltk SnowballStemmer('russian')"""
    if not isinstance(text, str) or not text:
        return []
    tokens = TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens if len(t) > 1 and t not in STOPWORDS]


# ----------------------------------------------------------------------------------
# BM25: строим вес-матрицу один раз, дальше скор запроса = сумма idf по колонкам,
# соответствующим токенам запроса (без пересчёта всей матрицы на каждый запрос).
# ----------------------------------------------------------------------------------
# Считает BM25-веса для матрицы счётчиков термов X (документы x термы).
# Вес документа по терму: tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl)); idf: ln((N-df+0.5)/(df+0.5)+1).
# Выход: (разреженная матрица весов W, вектор idf).
def build_bm25_weights(X, k1=BM25_K1, b=BM25_B):
    X = X.tocsr()
    n_docs = X.shape[0]
    doc_len = np.asarray(X.sum(axis=1)).ravel()
    avgdl = doc_len.mean() if n_docs else 0.0

    df = np.diff(X.tocsc().indptr)  # число документов, где встречается термин
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    row_idx = np.repeat(np.arange(n_docs), np.diff(X.indptr))
    tf = X.data.astype(np.float64)
    dl = doc_len[row_idx]
    denom = tf + k1 * (1.0 - b + b * dl / avgdl)
    weighted = tf * (k1 + 1.0) / denom
    W = sparse.csr_matrix((weighted, X.indices, X.indptr), shape=X.shape)
    return W, idf


def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Расстояние в км от одной точки (lat1, lon1) до массива точек. NaN на входе
    (нет координат) корректно даёт NaN на выходе - такие объявления просто не попадут
    ни в один радиус-шаг ниже (сравнение с NaN всегда False)."""
    R = 6371.0
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2_arr), np.radians(lon2_arr)
    dphi = lat2r - lat1r
    dlmb = lon2r - lon1r
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_location_coords(item_location_arr, item_lat_arr, item_lon_arr):
    """location_id -> (lat, lon), средняя координата объявлений в этой локации.
    Единственный доступный источник координат - сторона объявлений (у запросов есть
    только location_id, без lat/lon), поэтому это же используем, чтобы приближённо
    получить координаты и для search_location_id (те же id, то же пространство)."""
    valid = (
        (item_location_arr >= 0)
        & ~np.isnan(item_lat_arr.astype(float))
        & ~np.isnan(item_lon_arr.astype(float))
    )
    if not valid.any():
        return {}
    tmp = pd.DataFrame({
        "loc": item_location_arr[valid],
        "lat": item_lat_arr[valid],
        "lon": item_lon_arr[valid],
    })
    grouped = tmp.groupby("loc")[["lat", "lon"]].mean()
    return {int(loc): (row["lat"], row["lon"]) for loc, row in grouped.iterrows()}


# Достраивает координаты для search_location_id, которых нет среди объявлений корпуса
# (медиана координат выбранных в train объявлений). Локации с большим разбросом пропускаются.
# Изменяет corpus in-place (location_coords, location_spread). Выход: сколько локаций добавлено.
def extend_location_coords(corpus, train_path, exclude_texts=None, min_rows=3, max_spread_km=None):
    """Часть search_location_id (по бенчмарку ~17% запросов) вообще не встречается среди
    item_location_id корпуса. Для них нет координат, и
    без них не работают ни бонус за близость, ни сужение (recall на таких запросах на
    holdout был 0.43 против 0.87-0.96 на остальных). Достраиваем: координата локации =
    медиана координат объявлений, которые в train реально выбирали при поиске из неё;
    разброс (медианное расстояние до этой точки) используем как масштаб бонуса.
    exclude_texts - тексты holdout-запросов (чтобы валидация была без утечки)."""
    corpus.setdefault("location_spread", {})
    coords, spread_d = corpus["location_coords"], corpus["location_spread"]
    try:
        df = pd.read_parquet(train_path, columns=["search_query", "search_location_id",
                                                  "item_latitude", "item_longitude"])
    except Exception as e:  # нет колонок координат в train -- просто пропускаем
        print(f"[warn] extend_location_coords пропущено: {e}")
        return 0
    if exclude_texts:
        q = df["search_query"].astype(str).str.strip().str.lower()
        df = df[~q.isin(exclude_texts)]
    loc = pd.to_numeric(df["search_location_id"], errors="coerce")
    lat, lon = to_float(df["item_latitude"]), to_float(df["item_longitude"])
    ok = loc.notna() & lat.notna() & lon.notna() & ~loc.isin(list(coords.keys()))
    if not ok.any():
        return 0
    t = pd.DataFrame({"loc": loc[ok].astype(int).to_numpy(), "lat": lat[ok].to_numpy(), "lon": lon[ok].to_numpy()})
    cent = t.groupby("loc")[["lat", "lon"]].median()
    cent = cent[t.groupby("loc").size().reindex(cent.index) >= min_rows]
    if cent.empty:
        return 0
    t = t[t["loc"].isin(cent.index)]
    d = haversine_km(t["loc"].map(cent["lat"]).to_numpy(), t["loc"].map(cent["lon"]).to_numpy(),
                     t["lat"].to_numpy(), t["lon"].to_numpy())
    spread = pd.Series(d, index=t.index).groupby(t["loc"]).median()
    limit = EXTEND_MAX_SPREAD_KM if max_spread_km is None else max_spread_km
    added = 0
    for l, row in cent.iterrows():
        if float(spread[l]) > limit:
            continue
        coords[int(l)] = (float(row["lat"]), float(row["lon"]))
        spread_d[int(l)] = float(spread[l])
        added += 1
    return added


# Текстовый скор для всех документов: сумма idf*вес по колонкам термов запроса.
# Работает по CSC-массивам (indices/data/indptr) -- без пересчёта матрицы на каждый запрос.
# Выход: np.ndarray[n_items] со скором каждого объявления.
def score_terms(term_ids, idf, W_indices, W_data, W_indptr, n_items):
    scores = np.zeros(n_items, dtype=np.float64)
    for tid in term_ids:
        start, end = W_indptr[tid], W_indptr[tid + 1]
        rows = W_indices[start:end]
        if rows.size == 0:
            continue
        scores[rows] += idf[tid] * W_data[start:end]
    return scores


# ----------------------------------------------------------------------------------
# Мягкий скор совпадения фильтров -- версия для батч-обработки с предпосчитанным
# разбором item_infm_params_text (чтобы не парсить один и тот же item заново на
# каждый запрос, где он оказался кандидатом).
# ----------------------------------------------------------------------------------
# Быстрая версия soft_match_score: работает с уже разобранными фильтрами объявления
# (item_filters) и нормализованными текстами, чтобы не парсить объявление на каждый запрос.
# Выход: доля совпавших мягких фильтров запроса (0..1).
def soft_match_fast(search_filters, item_filters, item_text_norm, item_desc_norm):
    if not search_filters:
        return 1.0
    total = 0
    matched = 0
    for label, values in search_filters.items():
        if label in HARD_LABELS or label.startswith("Рейтинг пользователя"):
            continue
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
# lookup-слой: точный текст запроса из train -> набор item_id, присутствующих
# в benchmark_items.
# ----------------------------------------------------------------------------------
# Lookup-слой: нормализованный текст запроса из train -> множество item_id, присутствующих в корпусе.
# exclude_texts -- тексты holdout-запросов, исключаются, чтобы не было утечки при валидации.
# Выход: defaultdict(set).
def build_lookup(train_path, valid_item_ids, exclude_texts=None):
    """exclude_texts: множество нормализованных текстов запросов, которые нужно
    исключить при построении lookup -- используется в validate_holdout.py, чтобы
    holdout-запросы не "подсматривали" свой же правильный ответ через lookup-слой."""
    df = pd.read_parquet(train_path, columns=["search_query", "item_id"])
    df["search_query"] = df["search_query"].astype(str).str.strip().str.lower()
    df = df[df["item_id"].isin(valid_item_ids)]
    if exclude_texts:
        df = df[~df["search_query"].isin(exclude_texts)]
    lookup = defaultdict(set)
    for q, iid in zip(df["search_query"].to_numpy(), df["item_id"].to_numpy()):
        lookup[q].add(iid)
    del df
    gc.collect()
    return lookup


# ----------------------------------------------------------------------------------
# Загрузка и подготовка корпуса объявлений
# ----------------------------------------------------------------------------------
# Загружает benchmark_items и готовит всё, что нужно для поиска: числовые массивы (категория, рейтинг,
# отзывы, локация, координаты), текст документов для BM25 (заголовок x TITLE_REPEAT + параметры + описание),
# разобранные infm-фильтры объявлений, инвертированный индекс по "Вид услуги" и regexp меток.
# Выход: словарь corpus со всеми этими структурами.
def load_items(items_path, labels_path):
    cols = [
        "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
        "item_category_id", "item_rating", "item_rating_reviews_count", "item_location_id",
        "item_latitude", "item_longitude",
    ]
    items = pd.read_parquet(items_path, columns=cols)

    item_ids = items["item_id"].astype(str).to_numpy()
    item_category_arr = pd.to_numeric(items["item_category_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    item_rating_arr = to_float(items["item_rating"]).to_numpy()
    item_reviews_arr = to_float(items["item_rating_reviews_count"]).fillna(0).to_numpy()
    item_location_arr = pd.to_numeric(items["item_location_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    item_lat_arr = to_float(items["item_latitude"]).to_numpy()
    item_lon_arr = to_float(items["item_longitude"]).to_numpy()
    location_coords = build_location_coords(item_location_arr, item_lat_arr, item_lon_arr)

    title = items["item_title_raw"].fillna("").astype(str)
    infm = items["item_infm_params_text"].fillna("").astype(str)
    desc = items["item_description_raw"].fillna("").astype(str)

    doc_text = title
    for _ in range(TITLE_REPEAT - 1):
        doc_text = doc_text + " " + title
    doc_text = doc_text + " " + infm + " " + desc

    item_text_norm = (title + " " + infm + " " + desc).str.lower().to_numpy()
    item_desc_norm = desc.str.lower().to_numpy()
    infm_arr = infm.to_numpy()

    del items, title, desc
    gc.collect()

    labels = load_labels(labels_path)
    pattern = build_label_pattern(labels)

    n = len(item_ids)
    item_vid_sets = [None] * n
    item_infm_parsed = [None] * n
    for i in range(n):
        filters, _ = parse_infm_params(infm_arr[i], pattern)
        item_infm_parsed[i] = filters
        if "Вид услуги" in filters:
            vals = filters["Вид услуги"]
            item_vid_sets[i] = frozenset(normalize(v) for v in vals) if vals else frozenset()
    del infm, infm_arr
    gc.collect()

    label_inverted = defaultdict(list)
    no_label_indices = []
    for i, vset in enumerate(item_vid_sets):
        if vset is None:
            no_label_indices.append(i)
        elif vset:
            for v in vset:
                label_inverted[v].append(i)
    no_label_indices = np.array(no_label_indices, dtype=int)

    return {
        "item_ids": item_ids,
        "item_category_arr": item_category_arr,
        "item_rating_arr": item_rating_arr,
        "item_reviews_arr": item_reviews_arr,
        "item_location_arr": item_location_arr,
        "item_lat_arr": item_lat_arr,
        "item_lon_arr": item_lon_arr,
        "location_coords": location_coords,
        "item_text_norm": item_text_norm,
        "item_desc_norm": item_desc_norm,
        "item_vid_sets": item_vid_sets,
        "item_infm_parsed": item_infm_parsed,
        "label_inverted": label_inverted,
        "no_label_indices": no_label_indices,
        "doc_text": doc_text,
        "pattern": pattern,
    }


# Строит BM25-индекс: CountVectorizer с собственным токенизатором -> веса -> CSC-массивы.
# Выход: (vectorizer, idf, W_indices, W_data, W_indptr).
def build_bm25_index(doc_text):
    vectorizer = CountVectorizer(analyzer=tokenize, min_df=MIN_DF, max_features=MAX_FEATURES)
    try:
        X = vectorizer.fit_transform(doc_text)
    except ValueError:
        # На реальных 189k объявлений min_df=2 не должен обнулять словарь, но
        # подстраховываемся: если корпус аномально маленький/разреженный и после
        # обрезки не осталось ни одного термина -- пробуем без обрезки.
        vectorizer = CountVectorizer(analyzer=tokenize, min_df=1, max_features=MAX_FEATURES)
        X = vectorizer.fit_transform(doc_text)
    W, idf = build_bm25_weights(X)
    del X
    gc.collect()
    W_csc = W.tocsc()
    W_indices, W_data, W_indptr = W_csc.indices, W_csc.data, W_csc.indptr
    del W, W_csc
    gc.collect()
    return vectorizer, idf, W_indices, W_data, W_indptr


# ----------------------------------------------------------------------------------
# Кэш "Вид услуги"-масок по значению (одно и то же значение часто встречается
# во многих запросах -- считаем маску по всему корпусу один раз на значение).
# ----------------------------------------------------------------------------------
# Кэш булевых масок по значению "Вид услуги": объявления, у которых это значение указано,
# плюс объявления без метки, где значение встречается в сыром тексте (мягкий фолбэк).
class VidMaskCache:
    # Сохраняет ссылки на данные корпуса и создаёт пустой кэш.
    def __init__(self, item_vid_sets, label_inverted, no_label_indices, item_text_norm, n_items):
        self.item_vid_sets = item_vid_sets
        self.label_inverted = label_inverted
        self.no_label_indices = no_label_indices
        self.item_text_norm = item_text_norm
        self.n_items = n_items
        self.cache = {}

    # Возвращает булеву маску по корпусу для значения value (считается один раз, затем берётся из кэша).
    def get(self, value):
        if value in self.cache:
            return self.cache[value]
        mask = np.zeros(self.n_items, dtype=bool)
        idxs = self.label_inverted.get(value)
        if idxs:
            mask[idxs] = True
        for i in self.no_label_indices:
            if value in self.item_text_norm[i]:
                mask[i] = True
        self.cache[value] = mask
        return mask


# ----------------------------------------------------------------------------------
# Сборка кандидатов для одного запроса
# ----------------------------------------------------------------------------------
# Объединяет маски логическим И; None-маски пропускаются. Если масок нет -- вернёт None (нет ограничений).
def combine_masks(*masks):
    result = None
    for m in masks:
        if m is None:
            continue
        result = m.copy() if result is None else (result & m)
    return result


# Добор до top_k самыми популярными (по числу отзывов) объявлениями той же категории;
# при search_category == 0 -- по всему корпусу. Дубликаты уже выбранных пропускаются.
def fill_with_popular(chosen, chosen_set, item_ids, item_category_arr, popularity_arr, search_category, top_k):
    if search_category != 0:
        cand_idx = np.where(item_category_arr == search_category)[0]
    else:
        cand_idx = np.arange(len(item_ids))
    if cand_idx.size == 0:
        return chosen
    order = cand_idx[np.argsort(-popularity_arr[cand_idx])]
    for idx in order:
        iid = item_ids[idx]
        if iid in chosen_set:
            continue
        chosen.append(iid)
        chosen_set.add(iid)
        if len(chosen) >= top_k:
            break
    return chosen


# Главная функция: собирает до TOP_K кандидатов для одного запроса.
# Шаги:
#   1) BM25-скор по тексту запроса + текст фильтров;
#   2) плавный бонус за близость по локации (кроме delivery-запросов);
#   3) маски жёсткого фильтра: категория, рейтинг, "Вид услуги";
#   4) при большом числе кандидатов -- адаптивное сужение по локации (доп. приоритетный тир);
#   5) сначала кандидаты из lookup, затем тиры от строгого фильтра к нестрогому
#      (внутри тира -- пул POOL_SIZE лучших, ре-ранжирование скор + SOFT_MATCH_WEIGHT * мягкий скор);
#   6) добор популярными, если всё ещё < TOP_K.
# Выход: list[str] из item_id (не более TOP_K), порядок детерминирован.
def assemble_candidates_for_query(
    search_query, search_infm, search_category, search_location, is_delivery,
    corpus, vectorizer, idf, W_indices, W_data, W_indptr, vid_cache, lookup,
):
    item_ids = corpus["item_ids"]
    n_items = len(item_ids)

    q_tokens = tokenize((search_query or "") + " " + (search_infm or ""))
    vocab = vectorizer.vocabulary_
    term_ids = sorted({vocab[t] for t in q_tokens if t in vocab})
    text_score = score_terms(term_ids, idf, W_indices, W_data, W_indptr, n_items)

    # --- Локационный бонус: плавный по расстоянию, а не бинарный ------------------
    # exp(-dist/PROXIMITY_DECAY_KM) даёт полный LOCATION_BONUS при dist=0 (точное
    # совпадение location_id) и гладко убывает с расстоянием -- иначе при большом числе
    # текстово-идентичных кандидатов (частые запросы вроде "маникюр") все НЕ-точные
    # совпадения получали одинаковый нулевой бонус, и порядок среди них решался
    # произвольным тай-брейком по item_id, а не тем, кто реально ближе.
    dist_from_search = None
    if not is_delivery and search_location is not None and not pd.isna(search_location):
        coords = corpus["location_coords"].get(int(search_location))
        if coords is not None:
            dist_from_search = haversine_km(coords[0], coords[1], corpus["item_lat_arr"], corpus["item_lon_arr"])
            decay = max(PROXIMITY_DECAY_KM, min(corpus.get("location_spread", {}).get(int(search_location), 0.0), MAX_LOCATION_SPREAD_KM))
            proximity = np.where(np.isnan(dist_from_search), 0.0, np.exp(-dist_from_search / decay))
            text_score = text_score + LOCATION_BONUS * proximity
        else:
            # координаты этой локации не встречались среди объявлений -- fallback на
            # бинарное совпадение id, как раньше
            text_score = text_score + LOCATION_BONUS * (corpus["item_location_arr"] == search_location)

    threshold = extract_rating_threshold(search_infm or "")
    rmask = None
    if threshold is not None:
        rmask = np.isnan(corpus["item_rating_arr"]) | (corpus["item_rating_arr"] >= threshold)

    cmask = None
    if search_category != 0:
        cmask = corpus["item_category_arr"] == search_category

    search_filters, _ = parse_infm_params(search_infm or "", corpus["pattern"])
    vmask = None
    if search_filters.get("Вид услуги"):
        values = {normalize(v) for v in search_filters["Вид услуги"]}
        vmask = np.zeros(n_items, dtype=bool)
        for v in values:
            vmask |= vid_cache.get(v)

    strict_mask = combine_masks(cmask, rmask, vmask)

    tiers_masks = [
        strict_mask,
        combine_masks(cmask, rmask),
        combine_masks(cmask),
        None,
    ]

    # --- Адаптивное сужение по локации ---------------------------------------------
    # Если после текста + жёсткого фильтра кандидатов МНОГО -- это типовой/частый запрос
    # (см. LOCATION_NARROW_TRIGGER выше), и по нашим данным на train для таких запросов
    # правильный ответ почти всегда рядом. Сужаем прогрессивно (сперва тот же
    # location_id, потом расширяем радиус), и если набрали разумный пул -- вставляем его
    # ПЕРВЫМ тиром (после lookup), не удаляя старые тиры: они остаются страховкой, если
    # сужение почему-то не наберёт достаточно кандидатов.
    narrow_mask = None
    if not is_delivery and search_location is not None and not pd.isna(search_location):
        base_mask = strict_mask if strict_mask is not None else np.ones(n_items, dtype=bool)
        has_text = text_score > 0
        candidate_count = int(np.sum(base_mask & has_text))
        if candidate_count > LOCATION_NARROW_TRIGGER:
            same_loc = base_mask & has_text & (corpus["item_location_arr"] == search_location)
            if same_loc.sum() >= LOCATION_NARROW_MIN_POOL:
                narrow_mask = same_loc
            elif dist_from_search is not None:
                for radius in LOCATION_RADIUS_STEPS_KM:
                    candidate_mask = base_mask & has_text & (dist_from_search <= radius)
                    if candidate_mask.sum() >= LOCATION_NARROW_MIN_POOL:
                        narrow_mask = candidate_mask
                        break
                if narrow_mask is None:
                    # даже макс. радиус не набрал MIN_POOL -- берём что есть на макс.
                    # радиусе (лучше не пустой приоритетный тир, чем совсем никакой)
                    widest = base_mask & has_text & (dist_from_search <= LOCATION_RADIUS_STEPS_KM[-1])
                    if widest.sum() > 0:
                        narrow_mask = widest
            # если координаты search_location неизвестны (dist_from_search is None) --
            # не сужаем, работаем как раньше через обычные тиры

    if narrow_mask is not None:
        tiers_masks = [narrow_mask] + tiers_masks

    chosen = []
    chosen_set = set()

    lk = lookup.get(normalize(search_query or ""))
    if lk:
        for iid in sorted(lk):
            if iid not in chosen_set:
                chosen.append(iid)
                chosen_set.add(iid)
                if len(chosen) >= TOP_K:
                    break

    for mask in tiers_masks:
        if len(chosen) >= TOP_K:
            break
        scores = text_score if mask is None else np.where(mask, text_score, -np.inf)
        need = TOP_K - len(chosen)
        pool_n = min(n_items, max(POOL_SIZE, need * 4))
        if pool_n < n_items:
            top_idx = np.argpartition(scores, -pool_n)[-pool_n:]
        else:
            top_idx = np.arange(n_items)
        # kind="stable" -- чтобы порядок при точных совпадениях скора не зависел
        # от рандомизации хэшей/порядка обхода между разными запусками процесса
        top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]

        rescored = []
        for idx in top_idx:
            if scores[idx] == -np.inf:
                continue
            iid = item_ids[idx]
            if iid in chosen_set:
                continue
            soft = soft_match_fast(
                search_filters, corpus["item_infm_parsed"][idx],
                corpus["item_text_norm"][idx], corpus["item_desc_norm"][idx],
            )
            rescored.append((scores[idx] + SOFT_MATCH_WEIGHT * soft, iid))
        # вторичный ключ (iid) -- детерминированный тай-брейк при точном равенстве скора
        rescored.sort(key=lambda x: (-x[0], x[1]))

        for _, iid in rescored:
            if iid in chosen_set:
                continue
            chosen.append(iid)
            chosen_set.add(iid)
            if len(chosen) >= TOP_K:
                break

    if len(chosen) < TOP_K:
        chosen = fill_with_popular(
            chosen, chosen_set, item_ids, corpus["item_category_arr"],
            corpus["item_reviews_arr"], search_category, TOP_K,
        )

    return chosen[:TOP_K]


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
# Полный прогон: загрузка корпуса -> BM25-индекс -> lookup из train -> кандидаты для каждого запроса бенчмарка
# -> сохранение answer.csv (query_id, answer = item_id через пробел). Проверяет уникальность query_id.
def run(train_path, queries_path, items_path, labels_path, output_path, progress_every=200):
    t0 = time.time()

    # Печать сообщения с временной меткой от старта.
    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    log(f"стемминг: {'nltk snowball' if STEMMING_ENABLED else 'отключён (nltk не найден)'}")

    log("загружаю объявления...")
    corpus = load_items(items_path, labels_path)
    log(f"объявлений: {len(corpus['item_ids'])}")
    if EXTEND_LOCATIONS:
        n_ext = extend_location_coords(corpus, train_path)
        log(f"достроено координат для search_location_id вне корпуса: {n_ext}")

    log("строю BM25-индекс...")
    vectorizer, idf, W_indices, W_data, W_indptr = build_bm25_index(corpus["doc_text"])
    del corpus["doc_text"]
    gc.collect()
    log(f"индекс готов, термов: {len(idf)}")

    vid_cache = VidMaskCache(
        corpus["item_vid_sets"], corpus["label_inverted"], corpus["no_label_indices"],
        corpus["item_text_norm"], len(corpus["item_ids"]),
    )

    log("загружаю запросы бенчмарка...")
    q_cols = [
        "query_id", "search_query", "search_location_id", "search_is_delivery_search",
        "search_infm_params_text", "search_category",
    ]
    queries = pd.read_parquet(queries_path, columns=q_cols)
    log(f"запросов: {len(queries)}")
    unk = (~queries["search_location_id"].isin(list(corpus["location_coords"].keys()))).mean()
    log(f"доля запросов с неизвестными координатами локации: {unk:.3f}")

    log("строю lookup-слой из train...")
    lookup = build_lookup(train_path, set(corpus["item_ids"].tolist()))
    log(f"lookup построен, уникальных текстов запросов: {len(lookup)}")

    out_rows = []
    n = len(queries)
    for k, row in enumerate(queries.itertuples(index=False)):
        search_query = row.search_query if isinstance(row.search_query, str) else ""
        search_infm = row.search_infm_params_text if isinstance(row.search_infm_params_text, str) else ""
        search_location = row.search_location_id if pd.notna(row.search_location_id) else None
        is_delivery = bool(row.search_is_delivery_search == 1)
        search_category = int(row.search_category) if pd.notna(row.search_category) else 0

        cands = assemble_candidates_for_query(
            search_query, search_infm, search_category, search_location, is_delivery,
            corpus, vectorizer, idf, W_indices, W_data, W_indptr, vid_cache, lookup,
        )
        out_rows.append({"query_id": str(row.query_id), "answer": " ".join(cands)})

        if (k + 1) % progress_every == 0 or (k + 1) == n:
            log(f"обработано {k + 1}/{n} запросов")

    log("сохраняю ответ...")
    out_df = pd.DataFrame(out_rows, columns=["query_id", "answer"])
    assert out_df["query_id"].is_unique, "есть повторяющиеся query_id в ответе!"
    assert len(out_df) == len(queries), "число строк ответа не совпадает с числом запросов!"
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    log(f"готово: {output_path}")
    return out_df


# Аргументы командной строки: пути к файлам данных и файлу ответа.
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="train.parquet")
    p.add_argument("--queries", default="benchmark_queries.parquet")
    p.add_argument("--items", default="benchmark_items.parquet")
    p.add_argument("--labels", default="nazvaniya_filtrov.csv")
    p.add_argument("--output", default="answer.csv")
    p.add_argument("--progress-every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.train, args.queries, args.items, args.labels, args.output, args.progress_every)
