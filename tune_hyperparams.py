"""
tune_hyperparams.py

Подбор гиперпараметров candidates_pipeline.py на holdout из train.

Как устроено (чтобы не тратить время и не подогнаться под шум):
  * корпус, BM25-индекс, lookup и holdout-выборки строятся ОДИН раз;
  * между испытаниями меняются только параметры (для k1/b пересобираются лишь веса
    BM25 из уже посчитанной матрицы счётчиков -- это быстро);
  * holdout делится на две НЕПЕРЕСЕКАЮЩИЕ части: TUNE (по ней ищем) и CONFIRM
    (на ней проверяем лучшие конфиги -- защита от подгонки под случайность);
  * фаза 1: случайный поиск, фаза 2: локальный поиск вокруг лучшего;
  * каждый результат сразу пишется в CSV-лог -- при обрыве ничего не теряется.

Запуск:
    python tune_hyperparams.py --train train.parquet --items benchmark_items.parquet \
        --labels nazvaniya_filtrov.csv --n-tune 1000 --n-confirm 1000 \
        --n-random 30 --n-local 20 --time-budget-min 90

Итог: в конце печатается блок констант, который нужно вставить в candidates_pipeline.py
(и сохраняется best_params.json). Берите новый конфиг, только если на CONFIRM он лучше
базового больше чем на ~1.5-2 п.п. (стандартная ошибка при n=1000 около 1.1 п.п.).
"""

import argparse
import csv
import gc
import json
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import candidates_pipeline as cp
import validate_holdout as vh

BASE_RADIUS_STEPS = list(cp.LOCATION_RADIUS_STEPS_KM)

# name -> (kind, ...)   kind: log / lin / choice
SPACE = {
    "LOCATION_BONUS": ("log", 0.2, 6.0),
    "PROXIMITY_DECAY_KM": ("log", 3.0, 80.0),
    "SOFT_MATCH_WEIGHT": ("lin", 0.0, 4.0),
    "LOCATION_NARROW_MIN_POOL": ("choice", [20, 30, 40, 50, 75, 100, 150]),
    "POOL_SIZE": ("choice", [200, 400, 800]),
    "RADIUS_SCALE": ("choice", [0.5, 1.0, 2.0]),
    "BM25_K1": ("choice", [0.9, 1.2, 1.5, 2.0]),
    "BM25_B": ("choice", [0.5, 0.75, 0.9]),
}


# Текущие значения гиперпараметров из candidates_pipeline.py -- базовая точка сравнения.
def baseline_params():
    return {
        "LOCATION_BONUS": cp.LOCATION_BONUS,
        "PROXIMITY_DECAY_KM": cp.PROXIMITY_DECAY_KM,
        "SOFT_MATCH_WEIGHT": cp.SOFT_MATCH_WEIGHT,
        "LOCATION_NARROW_MIN_POOL": cp.LOCATION_NARROW_MIN_POOL,
        "POOL_SIZE": cp.POOL_SIZE,
        "RADIUS_SCALE": 1.0,
        "BM25_K1": cp.BM25_K1,
        "BM25_B": cp.BM25_B,
    }


# Случайное значение одного параметра по его описанию в SPACE (log / lin / choice).
def sample_value(spec, rng):
    kind = spec[0]
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
    if kind == "lin":
        return float(rng.uniform(spec[1], spec[2]))
    return spec[1][int(rng.integers(len(spec[1])))]


# Случайный полный набор гиперпараметров (фаза 1 -- случайный поиск).
def sample_params(rng):
    return {k: sample_value(v, rng) for k, v in SPACE.items()}


# Небольшое возмущение 1-2 параметров лучшего конфига (фаза 2 -- локальный поиск).
def perturb(p, rng):
    q = dict(p)
    names = list(SPACE)
    for k in rng.choice(names, size=int(rng.integers(1, 3)), replace=False):
        spec = SPACE[k]
        if spec[0] == "log":
            q[k] = float(np.clip(q[k] * np.exp(rng.normal(0, 0.35)), spec[1], spec[2]))
        elif spec[0] == "lin":
            q[k] = float(np.clip(q[k] + rng.normal(0, 0.15 * (spec[2] - spec[1])), spec[1], spec[2]))
        else:
            q[k] = spec[1][int(rng.integers(len(spec[1])))]
    return q


# BM25-индекс, где матрица счётчиков считается один раз, а веса пересобираются под (k1, b).
class BM25Index:
    """Счётчики считаются один раз; веса пересобираются под конкретные (k1, b)."""

    # Строит CountVectorizer и матрицу счётчиков термов по текстам документов.
    def __init__(self, doc_text):
        vec = CountVectorizer(analyzer=cp.tokenize, min_df=cp.MIN_DF, max_features=cp.MAX_FEATURES)
        try:
            self.X = vec.fit_transform(doc_text)
        except ValueError:
            vec = CountVectorizer(analyzer=cp.tokenize, min_df=1, max_features=cp.MAX_FEATURES)
            self.X = vec.fit_transform(doc_text)
        self.vectorizer = vec
        self.cur = None

    # Пересобирает BM25-веса для новых (k1, b); если параметры не менялись -- ничего не делает.
    def set(self, k1, b):
        if self.cur == (k1, b):
            return
        W, idf = cp.build_bm25_weights(self.X, k1, b)
        Wc = W.tocsc()
        self.idf, self.W_indices, self.W_data, self.W_indptr = idf, Wc.indices, Wc.data, Wc.indptr
        self.cur = (k1, b)
        del W, Wc
        gc.collect()


# Записывает набор параметров в константы модуля candidates_pipeline и обновляет веса BM25.
def apply_params(p, index):
    cp.LOCATION_BONUS = p["LOCATION_BONUS"]
    cp.PROXIMITY_DECAY_KM = p["PROXIMITY_DECAY_KM"]
    cp.SOFT_MATCH_WEIGHT = p["SOFT_MATCH_WEIGHT"]
    cp.LOCATION_NARROW_MIN_POOL = int(p["LOCATION_NARROW_MIN_POOL"])
    cp.POOL_SIZE = int(p["POOL_SIZE"])
    cp.LOCATION_RADIUS_STEPS_KM = [s * p["RADIUS_SCALE"] for s in BASE_RADIUS_STEPS]
    index.set(p["BM25_K1"], p["BM25_B"])


# Превращает DataFrame holdout в список кортежей (запрос, фильтры, категория, локация, delivery, item_id).
def to_tuples(df):
    out = []
    for r in df.itertuples(index=False):
        out.append((
            r.search_query if isinstance(r.search_query, str) else "",
            r.search_infm_params_text if isinstance(r.search_infm_params_text, str) else "",
            int(r.search_category) if r.search_category == r.search_category else 0,
            r.search_location_id if r.search_location_id == r.search_location_id else None,
            bool(r.search_is_delivery_search == 1),
            str(r.item_id),
        ))
    return out


# Recall@50 конфига p на наборе строк rows: доля запросов, где верный item_id попал в кандидаты.
def evaluate(p, rows, ctx):
    apply_params(p, ctx["index"])
    ix = ctx["index"]
    hits = 0
    for q, infm, cat, loc, deliv, iid in rows:
        cands = cp.assemble_candidates_for_query(
            q, infm, cat, loc, deliv, ctx["corpus"], ix.vectorizer, ix.idf,
            ix.W_indices, ix.W_data, ix.W_indptr, ctx["vid_cache"], ctx["lookup"],
        )
        hits += iid in cands
    return hits / len(rows)


# Подбор гиперпараметров: TUNE/CONFIRM-разбиение holdout, случайный + локальный поиск,
# проверка лучших конфигов на CONFIRM, принятие нового конфига только при приросте выше шума (>1.5 ст. ошибок).
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="train.parquet")
    ap.add_argument("--items", default="benchmark_items.parquet")
    ap.add_argument("--labels", default="nazvaniya_filtrov.csv")
    ap.add_argument("--n-tune", type=int, default=1000)
    ap.add_argument("--n-confirm", type=int, default=1000)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--n-local", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5, help="сколько лучших конфигов проверить на CONFIRM")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-budget-min", type=float, default=0, help="0 = без лимита")
    ap.add_argument("--uniform-sample", action="store_true", help="обычная случайная выборка (по умолчанию -- под распределение бенчмарка)")
    ap.add_argument("--log", default="tuning_log.csv")
    ap.add_argument("--best-out", default="best_params.json")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # Печать сообщения с временной меткой (минуты от старта).
    def log(msg):
        print(f"[{(time.time() - t0) / 60:6.1f} мин] {msg}", flush=True)

    # True, если исчерпан лимит времени --time-budget-min.
    def out_of_time():
        return args.time_budget_min > 0 and (time.time() - t0) / 60 > args.time_budget_min

    log("загружаю корпус и строю индекс...")
    corpus = cp.load_items(args.items, args.labels)
    index = BM25Index(corpus.pop("doc_text"))
    gc.collect()
    vid_cache = cp.VidMaskCache(
        corpus["item_vid_sets"], corpus["label_inverted"], corpus["no_label_indices"],
        corpus["item_text_norm"], len(corpus["item_ids"]),
    )

    log("готовлю holdout (TUNE + CONFIRM, без пересечений)...")
    valid_ids = set(corpus["item_ids"].tolist())
    strata = None if args.uniform_sample else vh.BENCH_STRATA
    cat0 = 0.0 if args.uniform_sample else vh.BENCH_CAT0
    eval_rows, exclude_texts = vh.load_eval_rows(args.train, valid_ids, args.n_tune + args.n_confirm, args.seed, strata, cat0)
    tune_rows = to_tuples(eval_rows.iloc[: args.n_tune])
    confirm_rows = to_tuples(eval_rows.iloc[args.n_tune:])
    lookup = cp.build_lookup(args.train, valid_ids, exclude_texts=exclude_texts)
    if cp.EXTEND_LOCATIONS:
        cp.extend_location_coords(corpus, args.train, exclude_texts=exclude_texts)
    log(f"TUNE={len(tune_rows)}  CONFIRM={len(confirm_rows)}")

    ctx = {"corpus": corpus, "index": index, "vid_cache": vid_cache, "lookup": lookup}
    results = []  # (tune_score, params, phase)

    fields = ["phase", "tune_recall", "confirm_recall"] + list(SPACE)
    logf = open(args.log, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(logf, fieldnames=fields)
    writer.writeheader()

    # Записывает результат испытания в CSV-лог сразу (при обрыве ничего не теряется).
    def record(phase, p, tune_s, confirm_s=""):
        writer.writerow({"phase": phase, "tune_recall": round(tune_s, 4), "confirm_recall": confirm_s, **p})
        logf.flush()

    base = baseline_params()
    t1 = time.time()
    base_tune = evaluate(base, tune_rows, ctx)
    log(f"БАЗОВЫЙ конфиг: TUNE recall = {base_tune:.4f}  (одно испытание ~{time.time() - t1:.0f} сек)")
    results.append((base_tune, base, "baseline"))
    record("baseline", base, base_tune)

    log(f"фаза 1: случайный поиск ({args.n_random} испытаний)")
    for i in range(args.n_random):
        if out_of_time():
            log("лимит времени -- останавливаюсь")
            break
        p = sample_params(rng)
        s = evaluate(p, tune_rows, ctx)
        results.append((s, p, "random"))
        record("random", p, s)
        log(f"  random {i + 1}/{args.n_random}: {s:.4f}  (лучший пока {max(r[0] for r in results):.4f})")

    log(f"фаза 2: локальный поиск вокруг лучшего ({args.n_local} шагов)")
    for i in range(args.n_local):
        if out_of_time():
            log("лимит времени -- останавливаюсь")
            break
        best_s, best_p, _ = max(results, key=lambda r: r[0])
        p = perturb(best_p, rng)
        s = evaluate(p, tune_rows, ctx)
        results.append((s, p, "local"))
        record("local", p, s)
        mark = "  <-- новый лучший" if s > best_s else ""
        log(f"  local {i + 1}/{args.n_local}: {s:.4f}{mark}")

    log("проверка лучших конфигов на CONFIRM...")
    uniq, seen = [], set()
    for s, p, ph in sorted(results, key=lambda r: -r[0]):
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append((s, p, ph))
    finalists = uniq[: args.top_k]
    if not any(ph == "baseline" for _, _, ph in finalists):
        finalists.append(next(x for x in uniq if x[2] == "baseline"))

    scored = []
    for s, p, ph in finalists:
        c = evaluate(p, confirm_rows, ctx)
        scored.append((c, s, p, ph))
        record("confirm:" + ph, p, s, round(c, 4))
        log(f"  {ph:8s} TUNE={s:.4f}  CONFIRM={c:.4f}")
    logf.close()

    base_confirm = next(c for c, _, _, ph in scored if ph == "baseline")
    best_c, best_s, best_p, best_ph = max(scored, key=lambda x: x[0])
    se = float(np.sqrt(base_confirm * (1 - base_confirm) / len(confirm_rows)))
    gain = best_c - base_confirm

    print()
    print(f"CONFIRM: базовый = {base_confirm:.4f}, лучший = {best_c:.4f}  (прирост {gain * 100:+.2f} п.п., ст. ошибка ~{se * 100:.2f} п.п.)")
    if best_ph == "baseline" or gain < 1.5 * se:
        print("Прирост в пределах шума -- оставляйте текущие параметры.")
        best_p = base
    else:
        print("Прирост выше шума. Вставьте в candidates_pipeline.py:")
        print(f"  LOCATION_BONUS = {best_p['LOCATION_BONUS']:.3f}")
        print(f"  PROXIMITY_DECAY_KM = {best_p['PROXIMITY_DECAY_KM']:.2f}")
        print(f"  SOFT_MATCH_WEIGHT = {best_p['SOFT_MATCH_WEIGHT']:.3f}")
        print(f"  LOCATION_NARROW_MIN_POOL = {int(best_p['LOCATION_NARROW_MIN_POOL'])}")
        print(f"  POOL_SIZE = {int(best_p['POOL_SIZE'])}")
        print(f"  BM25_K1 = {best_p['BM25_K1']}")
        print(f"  BM25_B = {best_p['BM25_B']}")
        print(f"  LOCATION_RADIUS_STEPS_KM = {[round(s * best_p['RADIUS_SCALE'], 1) for s in BASE_RADIUS_STEPS]}")
    with open(args.best_out, "w", encoding="utf-8") as f:
        json.dump(best_p, f, ensure_ascii=False, indent=2)
    log(f"готово; лог: {args.log}, параметры: {args.best_out}")


if __name__ == "__main__":
    main()
