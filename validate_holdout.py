"""
validate_holdout.py

Прокси-валидация Recall@50 на train, БЕЗ обращения к настоящему
benchmark_queries.parquet (у него нет разметки). Идея:

  1. Берём НАСТОЯЩИЙ корпус benchmark_items.parquet (189k объявлений) -- именно
     в нём придётся искать на реальной отправке, а не в маленьком синтетическом
     корпусе.
  2. Из train.parquet оставляем только строки, у которых item_id ДЕЙСТВИТЕЛЬНО
     присутствует в этом корпусе (~9.6% строк, см. Анализ) -- иначе "правильный"
     ответ в принципе недостижим, и это была бы нечестная проверка.
  3. Сэмплируем из них N строк как псевдо-запросы с известным правильным
     ответом (используем ВСЕ признаки строки -- текст, локацию, категорию,
     фильтры -- ровно как реальный запрос).
  4. Строим lookup-слой из train, ИСКЛЮЧИВ тексты запросов holdout-выборки --
     иначе lookup-слой тривиально "угадывал" бы правильный ответ по своей же
     подсказке (утечка).
  5. Прогоняем ту же assemble_candidates_for_query(), что и в боевом пайплайне,
     и считаем, попал ли настоящий item_id в топ-50.

Дополнительно: разбивка Recall@50 по частоте текста запроса в train (частые
шаблонные запросы вроде "маникюр" vs редкие/специфичные) -- чтобы видеть,
работает ли конкретно локационное сужение там, где должно.

Запуск:
    python validate_holdout.py --train train.parquet --items benchmark_items.parquet \
        --labels nazvaniya_filtrov.csv --n-eval 1500

Подбор порогов (сравнить с выключенным сужением):
    python validate_holdout.py ... --disable-narrow
    python validate_holdout.py ... --location-narrow-trigger 150 --location-narrow-min-pool 150
"""

import argparse
import gc
import time

import numpy as np
import pandas as pd

import candidates_pipeline as cp


# доли групп по частоте текста запроса в бенчмарке (g0: текст не встречался в train,
# g1: 1-10 раз, g2: >10 раз) и доля запросов с search_category=0
BENCH_STRATA = {"g0": 0.63, "g1": 0.296, "g2": 0.074}
BENCH_CAT0 = 0.091


# Формирует holdout из train: только строки, где верный item_id есть в корпусе benchmark_items
# (иначе ответ недостижим). Опционально: стратификация по частоте текста запроса (strata) и доля
# запросов с search_category=0 (cat0_frac) под распределение бенчмарка.
# Выход: (eval_rows, exclude_texts) -- строки holdout и их тексты для исключения из lookup.
def load_eval_rows(train_path, valid_item_ids, n_eval, seed, strata=None, cat0_frac=0.0):
    cols = [
        "search_query", "search_location_id", "search_is_delivery_search",
        "search_infm_params_text", "search_category", "item_id",
    ]
    train = pd.read_parquet(train_path, columns=cols)

    # частота текста запроса -- считаем по ПОЛНОМУ train (это и есть "насколько
    # шаблонный/частый запрос" в реальности, а не только среди строк с валидным item_id)
    freq = train["search_query"].astype(str).str.strip().str.lower().value_counts()

    evaluable = train[train["item_id"].isin(valid_item_ids)].copy()
    print(f"строк train с item_id из benchmark_items: {len(evaluable)} / {len(train)} "
          f"({len(evaluable)/len(train):.1%})")

    evaluable["query_norm"] = evaluable["search_query"].astype(str).str.strip().str.lower()
    evaluable["freq"] = evaluable["query_norm"].map(freq).fillna(1).astype(int)

    if strata:
        # в holdout сама строка входит в freq: freq=1 <-> в бенчмарке "текст не встречался в train"
        g = np.where(evaluable["freq"] <= 1, "g0", np.where(evaluable["freq"] <= 11, "g1", "g2"))
        parts = []
        for name, w in strata.items():
            pool = evaluable[g == name]
            want = int(round(n_eval * w))
            if len(pool) < want:
                print(f"[warn] группа {name}: доступно {len(pool)} строк, нужно {want} -- беру все")
            parts.append(pool.sample(n=min(want, len(pool)), random_state=seed))
        eval_rows = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        n_eval = min(n_eval, len(evaluable))
        eval_rows = evaluable.sample(n=n_eval, random_state=seed).reset_index(drop=True)

    if cat0_frac > 0:
        rng = np.random.default_rng(seed)
        m = rng.random(len(eval_rows)) < cat0_frac
        eval_rows.loc[m, "search_category"] = 0
        print(f"search_category=0 выставлено у {m.sum()} строк ({m.mean():.1%}) -- без фильтра по категории")

    exclude_texts = set(eval_rows["query_norm"].unique().tolist())
    del train, evaluable
    gc.collect()
    return eval_rows, exclude_texts


# Относит частоту текста запроса в train к корзине для разбивки метрики ("1 раз", "2 раза", "3-4", "5-10", ">10").
def freq_bucket(n):
    if n <= 1:
        return "1 раз"
    if n <= 2:
        return "2 раза"
    if n <= 4:
        return "3-4"
    if n <= 10:
        return "5-10"
    return ">10"


# Прогоняет боевой assemble_candidates_for_query на holdout и считает proxy Recall@50
# (попал ли верный item_id в топ-50), с разбивкой по частоте запроса и по категории 0.
# Lookup строится без holdout-текстов (без утечки). dump_path -- CSV с hit/miss для разбора ошибок.
# Выход: DataFrame holdout-строк с колонками hit и bucket.
def run_validation(train_path, items_path, labels_path, n_eval, seed, progress_every=200, dump_path=None,
                   strata=None, cat0_frac=0.0):
    t0 = time.time()

    # Печать сообщения с временной меткой (секунды от старта).
    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    log("загружаю benchmark_items (реальный корпус)...")
    corpus = cp.load_items(items_path, labels_path)
    log(f"объявлений: {len(corpus['item_ids'])}")

    log("строю BM25-индекс...")
    vectorizer, idf, W_indices, W_data, W_indptr = cp.build_bm25_index(corpus["doc_text"])
    del corpus["doc_text"]
    gc.collect()
    log(f"индекс готов, термов: {len(idf)}")

    vid_cache = cp.VidMaskCache(
        corpus["item_vid_sets"], corpus["label_inverted"], corpus["no_label_indices"],
        corpus["item_text_norm"], len(corpus["item_ids"]),
    )

    log("готовлю holdout-выборку из train...")
    valid_item_ids = set(corpus["item_ids"].tolist())
    eval_rows, exclude_texts = load_eval_rows(train_path, valid_item_ids, n_eval, seed, strata, cat0_frac)
    log(f"holdout-запросов: {len(eval_rows)}")

    log("строю lookup-слой (без holdout-текстов, чтобы не было утечки)...")
    lookup = cp.build_lookup(train_path, valid_item_ids, exclude_texts=exclude_texts)
    log(f"lookup построен, уникальных текстов: {len(lookup)}")
    if cp.EXTEND_LOCATIONS:
        n_ext = cp.extend_location_coords(corpus, train_path, exclude_texts=exclude_texts)
        log(f"достроено координат для search_location_id вне корпуса: {n_ext}")

    hits = np.zeros(len(eval_rows), dtype=bool)
    n = len(eval_rows)
    for k, row in enumerate(eval_rows.itertuples(index=False)):
        search_query = row.search_query if isinstance(row.search_query, str) else ""
        search_infm = row.search_infm_params_text if isinstance(row.search_infm_params_text, str) else ""
        search_location = row.search_location_id if pd.notna(row.search_location_id) else None
        is_delivery = bool(row.search_is_delivery_search == 1)
        search_category = int(row.search_category) if pd.notna(row.search_category) else 0

        cands = cp.assemble_candidates_for_query(
            search_query, search_infm, search_category, search_location, is_delivery,
            corpus, vectorizer, idf, W_indices, W_data, W_indptr, vid_cache, lookup,
        )
        hits[k] = str(row.item_id) in cands

        if (k + 1) % progress_every == 0 or (k + 1) == n:
            log(f"обработано {k + 1}/{n}, промежуточный Recall@50 = {hits[:k+1].mean():.4f}")

    eval_rows["hit"] = hits
    eval_rows["bucket"] = eval_rows["freq"].apply(freq_bucket)

    if dump_path:
        pos = {iid: i for i, iid in enumerate(corpus["item_ids"])}
        idx = np.array([pos[str(x)] for x in eval_rows["item_id"]])
        d = eval_rows.drop(columns=["query_norm"]).copy()
        d["true_item_category"] = corpus["item_category_arr"][idx]
        d["true_item_location_id"] = corpus["item_location_arr"][idx]
        d["true_item_rating"] = corpus["item_rating_arr"][idx]
        d["true_item_text_head"] = [str(corpus["item_text_norm"][i])[:200].replace("\n", " ") for i in idx]
        dist = []
        for sl, i in zip(d["search_location_id"], idx):
            c = corpus["location_coords"].get(int(sl)) if pd.notna(sl) else None
            if c is None or np.isnan(corpus["item_lat_arr"][i]):
                dist.append(np.nan)
            else:
                dist.append(float(cp.haversine_km(c[0], c[1], corpus["item_lat_arr"][i:i+1], corpus["item_lon_arr"][i:i+1])[0]))
        d["dist_km_to_true_item"] = dist
        d.to_csv(dump_path, index=False, encoding="utf-8-sig")
        log(f"выгрузка holdout-строк (hit=True/False): {dump_path}")

    print()
    print(f"=== Recall@50 (proxy, n={len(eval_rows)}) = {hits.mean():.4f} ===")
    print()
    print("по частоте текста запроса в train (частый/шаблонный vs редкий):")
    order = ["1 раз", "2 раза", "3-4", "5-10", ">10"]
    g = eval_rows.groupby("bucket")["hit"].agg(["mean", "size"]).reindex(order)
    print(g.rename(columns={"mean": "recall@50", "size": "n"}).round(4).to_string())

    if (eval_rows["search_category"] == 0).any():
        c0 = eval_rows["search_category"] == 0
        print(f"\nкатегория 0 (n={c0.sum()}): recall = {eval_rows[c0].hit.mean():.4f};  остальные: {eval_rows[~c0].hit.mean():.4f}")
    return eval_rows


# Аргументы командной строки: данные, размер holdout и переопределения констант пайплайна для A/B-проверок.
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="train.parquet")
    p.add_argument("--items", default="benchmark_items.parquet")
    p.add_argument("--labels", default="nazvaniya_filtrov.csv")
    p.add_argument("--n-eval", type=int, default=1500, help="сколько строк train сэмплировать как holdout-запросы")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--dump", default=None, help="путь к CSV: выгрузить все holdout-строки с hit и признаками правильного объявления")

    # ручная настройка порогов для калибровки (перекрывают константы в candidates_pipeline)
    p.add_argument("--location-bonus", type=float, default=None)
    p.add_argument("--proximity-decay-km", type=float, default=None)
    p.add_argument("--location-narrow-trigger", type=int, default=None)
    p.add_argument("--location-narrow-min-pool", type=int, default=None)
    p.add_argument("--disable-narrow", action="store_true",
                    help="полностью выключить адаптивное сужение по локации (для A/B сравнения)")
    p.add_argument("--soft-match-weight", type=float, default=None)
    p.add_argument("--match-benchmark", action="store_true",
                    help="выборка под распределение бенчмарка: доли по частоте текста запроса + 9.1%% категории 0")
    p.add_argument("--cat0-frac", type=float, default=None)
    p.add_argument("--extend-max-spread", type=float, default=None,
                    help="порог разброса (км) для достроенных координат; 1e9 = без порога")
    p.add_argument("--no-extend-locations", action="store_true",
                    help="выключить достройку координат для неизвестных search_location_id (для A/B)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.location_bonus is not None:
        cp.LOCATION_BONUS = args.location_bonus
    if args.proximity_decay_km is not None:
        cp.PROXIMITY_DECAY_KM = args.proximity_decay_km
    if args.location_narrow_trigger is not None:
        cp.LOCATION_NARROW_TRIGGER = args.location_narrow_trigger
    if args.location_narrow_min_pool is not None:
        cp.LOCATION_NARROW_MIN_POOL = args.location_narrow_min_pool
    if args.soft_match_weight is not None:
        cp.SOFT_MATCH_WEIGHT = args.soft_match_weight
    if args.extend_max_spread is not None:
        cp.EXTEND_MAX_SPREAD_KM = args.extend_max_spread
    if args.no_extend_locations:
        cp.EXTEND_LOCATIONS = False
    if args.disable_narrow:
        cp.LOCATION_NARROW_TRIGGER = float("inf")

    print(f"конфигурация: LOCATION_BONUS={cp.LOCATION_BONUS} PROXIMITY_DECAY_KM={cp.PROXIMITY_DECAY_KM} "
          f"LOCATION_NARROW_TRIGGER={cp.LOCATION_NARROW_TRIGGER} LOCATION_NARROW_MIN_POOL={cp.LOCATION_NARROW_MIN_POOL} "
          f"SOFT_MATCH_WEIGHT={cp.SOFT_MATCH_WEIGHT}")
    print()

    strata = BENCH_STRATA if args.match_benchmark else None
    cat0 = args.cat0_frac if args.cat0_frac is not None else (BENCH_CAT0 if args.match_benchmark else 0.0)
    run_validation(args.train, args.items, args.labels, args.n_eval, args.seed, args.progress_every, args.dump, strata, cat0)
