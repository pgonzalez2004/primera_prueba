import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# =========================
# Config
# =========================

@dataclass
class Config:
    seed: int = 43
    start_date: date = date(2026, 5, 15)
    end_date: date = date(2026, 6, 16)
    ejemplar_ids: List[int] = None

    UBI_CLINICA: int = 1
    UBI_BEBEDERO: int = 2
    UBI_ENTRENAMIENTO: int = 3
    UBI_HENERAS: Tuple[int, int, int] = (13, 14, 19)

    tz = timezone.utc

    clinic_horses_per_day: Tuple[int, int] = (0, 3)
    clinic_duration_min: Tuple[int, int] = (15, 30)
    clinic_detections: Tuple[int, int] = (5, 10)
    clinic_window: Tuple[time, time] = (time(7, 30), time(16, 0))

    training_horses_per_day: Tuple[int, int] = (8, 10)
    training_duration_min: Tuple[int, int] = (20, 40)
    training_detections: Tuple[int, int] = (5, 10)
    training_window: Tuple[time, time] = (time(7, 30), time(16, 0))

    water_times_per_day: Tuple[int, int] = (4, 8)
    water_duration_sec: Tuple[int, int] = (40, 120)
    water_detections: Tuple[int, int] = (10, 30)
    water_window: Tuple[time, time] = (time(6, 0), time(0, 59))

    eat_times_per_day: Tuple[int, int] = (3, 6)
    eat_duration_min: Tuple[int, int] = (15, 90)
    eat_detections: Tuple[int, int] = (10, 150)
    eat_window: Tuple[time, time] = (time(6, 0), time(23, 59))


# =========================
# Generación de sesiones
# =========================

def fmt_dt(dt: datetime) -> str:
    s = dt.strftime("%Y-%m-%d %H:%M:%S%z")
    if s.endswith("+0000"):
        s = s[:-5] + "+00"
    return s


def rand_datetime_in_window(day: date, start_t: time, end_t: time, tz) -> datetime:
    start_dt = datetime.combine(day, start_t, tzinfo=tz)
    if (end_t.hour, end_t.minute, end_t.second) < (start_t.hour, start_t.minute, start_t.second):
        end_dt = datetime.combine(day + timedelta(days=1), end_t, tzinfo=tz)
    else:
        end_dt = datetime.combine(day, end_t, tzinfo=tz)

    total_seconds = int((end_dt - start_dt).total_seconds())
    offset = random.randint(0, max(total_seconds, 0))
    return start_dt + timedelta(seconds=offset)


def random_detections(det_range: Tuple[int, int], last_value: Optional[int] = None) -> int:
    lo, hi = det_range
    if lo >= hi:
        return lo
    if last_value is not None:
        for _ in range(6):
            v = random.randint(lo, hi)
            if v != last_value:
                return v
    return random.randint(lo, hi)


def generate_rows(cfg: Config) -> List[dict]:
    random.seed(cfg.seed)

    if cfg.ejemplar_ids is None:
        cfg.ejemplar_ids = list(range(200, 224))

    rows = []
    next_id = 1
    last_det: Dict[Tuple[int, int], int] = {}

    def add_event(ejemplar_id: int, ubi_id: int, start: datetime, duration_sec: int, det_range: Tuple[int, int]):
        nonlocal next_id
        end = start + timedelta(seconds=duration_sec)
        key = (ejemplar_id, ubi_id)
        det = random_detections(det_range, last_det.get(key))
        last_det[key] = det

        rows.append({
            "id": next_id,
            "entrada": fmt_dt(start),
            "salida": fmt_dt(end),
            "duracion": duration_sec,
            "detecciones": det,
            "ejemplar_id": ejemplar_id,
            "ubi_id": ubi_id,
        })
        next_id += 1

    day = cfg.start_date
    while day <= cfg.end_date:
        k = random.randint(cfg.clinic_horses_per_day[0], cfg.clinic_horses_per_day[1])
        clinic_horses = random.sample(cfg.ejemplar_ids, k=min(k, len(cfg.ejemplar_ids)))
        for hid in clinic_horses:
            start = rand_datetime_in_window(day, cfg.clinic_window[0], cfg.clinic_window[1], cfg.tz)
            dur_min = random.randint(*cfg.clinic_duration_min)
            add_event(hid, cfg.UBI_CLINICA, start, dur_min * 60, cfg.clinic_detections)

        k = random.randint(cfg.training_horses_per_day[0], cfg.training_horses_per_day[1])
        training_horses = random.sample(cfg.ejemplar_ids, k=min(k, len(cfg.ejemplar_ids)))
        for hid in training_horses:
            start = rand_datetime_in_window(day, cfg.training_window[0], cfg.training_window[1], cfg.tz)
            dur_min = random.randint(*cfg.training_duration_min)
            add_event(hid, cfg.UBI_ENTRENAMIENTO, start, dur_min * 60, cfg.training_detections)

        for hid in cfg.ejemplar_ids:
            n_water = random.randint(*cfg.water_times_per_day)
            for _ in range(n_water):
                start = rand_datetime_in_window(day, cfg.water_window[0], cfg.water_window[1], cfg.tz)
                dur_sec = random.randint(*cfg.water_duration_sec)
                add_event(hid, cfg.UBI_BEBEDERO, start, dur_sec, cfg.water_detections)

            n_eat = random.randint(*cfg.eat_times_per_day)
            for _ in range(n_eat):
                start = rand_datetime_in_window(day, cfg.eat_window[0], cfg.eat_window[1], cfg.tz)
                dur_min = random.randint(*cfg.eat_duration_min)
                ubi = random.choice(cfg.UBI_HENERAS)
                add_event(hid, ubi, start, dur_min * 60, cfg.eat_detections)

        day += timedelta(days=1)

    rows.sort(key=lambda r: (r["entrada"], r["id"]))
    for i, r in enumerate(rows, start=1):
        r["id"] = i
    return rows


# =========================
# Construcción de matrices
# =========================

def rows_to_df(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["entrada"] = pd.to_datetime(df["entrada"])
    df["salida"] = pd.to_datetime(df["salida"])
    return df


def construir_matriz(df: pd.DataFrame, tipo_matriz: str) -> pd.DataFrame:
    ejemplares = sorted(df["ejemplar_id"].unique())
    matriz = pd.DataFrame(0.0, index=ejemplares, columns=ejemplares)

    if tipo_matriz == "tiempo":
        for _, row in df.iterrows():
            e = int(row["ejemplar_id"])
            matriz.loc[e, e] += float(row["duracion"])
        return matriz

    if tipo_matriz == "coincidencia":
        for _, row in df.iterrows():
            e = int(row["ejemplar_id"])
            matriz.loc[e, e] += 1.0
        return matriz

    if tipo_matriz == "coincidencia_30":
        df2 = df[df["duracion"] > 30].copy()
        for _, row in df2.iterrows():
            e = int(row["ejemplar_id"])
            matriz.loc[e, e] += 1.0
        return matriz

    raise ValueError(f"Tipo de matriz no soportado: {tipo_matriz}")


def normalizar_matriz(matriz: pd.DataFrame, n_instancias: int = 30) -> pd.DataFrame:
    return matriz / n_instancias


# =========================
# Clustering y métricas
# =========================

def preparar_x(matriz: pd.DataFrame) -> pd.DataFrame:
    X = matriz.copy()
    X = X.fillna(0)
    return X


def evaluar_modelos(X: pd.DataFrame):
    resultados = []

    if len(X) < 2:
        return resultados

    X_scaled = StandardScaler().fit_transform(X)

    labels_kmeans = None
    labels_aggl = None
    labels_db = None

    ks = list(range(2, min(len(X), 15) + 1))

    mejor_kmeans = None
    mejor_aggl = None

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue

        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        dbi = davies_bouldin_score(X_scaled, labels)

        resultados.append({
            "modelo": "kmeans",
            "parametro": f"k={k}",
            "k": k,
            "eps": None,
            "silhouette": sil,
            "calinski": cal,
            "davies_bouldin": dbi,
            "labels": labels
        })

        if mejor_kmeans is None or cal > mejor_kmeans["calinski"]:
            mejor_kmeans = resultados[-1]

    for k in ks:
        for linkage in ["ward", "complete", "average"]:
            try:
                ag = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = ag.fit_predict(X_scaled)
                if len(set(labels)) < 2:
                    continue

                sil = silhouette_score(X_scaled, labels)
                cal = calinski_harabasz_score(X_scaled, labels)
                dbi = davies_bouldin_score(X_scaled, labels)

                resultados.append({
                    "modelo": f"agglomerative_{linkage}",
                    "parametro": f"k={k}",
                    "k": k,
                    "eps": None,
                    "silhouette": sil,
                    "calinski": cal,
                    "davies_bouldin": dbi,
                    "labels": labels
                })

                if mejor_aggl is None or cal > mejor_aggl["calinski"]:
                    mejor_aggl = resultados[-1]
            except Exception:
                pass

    for eps in [0.1, 0.2, 0.4]:
        db = DBSCAN(eps=eps, min_samples=30)
        labels = db.fit_predict(X_scaled)
        cluster_mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2 or cluster_mask.sum() < 2:
            resultados.append({
                "modelo": "dbscan",
                "parametro": f"eps={eps}",
                "k": n_clusters,
                "eps": eps,
                "silhouette": None,
                "calinski": None,
                "davies_bouldin": None,
                "labels": labels
            })
            continue

        sil = silhouette_score(X_scaled[cluster_mask], labels[cluster_mask])
        cal = calinski_harabasz_score(X_scaled[cluster_mask], labels[cluster_mask])
        dbi = davies_bouldin_score(X_scaled[cluster_mask], labels[cluster_mask])

        resultados.append({
            "modelo": "dbscan",
            "parametro": f"eps={eps}",
            "k": n_clusters,
            "eps": eps,
            "silhouette": sil,
            "calinski": cal,
            "davies_bouldin": dbi,
            "labels": labels
        })

    return resultados, mejor_kmeans, mejor_aggl


def elegir_mejor_modelo(resultados):
    validos = [r for r in resultados if r["silhouette"] is not None]
    if not validos:
        return None
    return sorted(validos, key=lambda x: (
        x["calinski"] if x["calinski"] is not None else -1,
        x["silhouette"] if x["silhouette"] is not None else -1,
        -x["davies_bouldin"] if x["davies_bouldin"] is not None else -1
    ), reverse=True)[0]


# =========================
# Experimento principal
# =========================

# ... (Mantén toda la parte superior de imports y configuración igual) ...

def ejecutar_experimento():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    resumen_filas = []

    for instancia in range(1, 31):
        cfg = Config(seed=instancia)
        rows = generate_rows(cfg)
        df = rows_to_df(rows)
        n_ejemplares = len(df["ejemplar_id"].unique())

        for tipo_matriz in ["tiempo", "coincidencia", "coincidencia_30"]:
            matriz = construir_matriz(df, tipo_matriz)
            matriz_norm = normalizar_matriz(matriz, n_instancias=30)
            X = preparar_x(matriz_norm)
            
            # Obtenemos los resultados de todos los modelos
            resultados, _, _ = evaluar_modelos(X)

            for res in resultados:
                # Filtrar solo modelos válidos o con resultados significativos
                resumen_filas.append({
                    "Instancia": instancia,
                    "Nº de ejemplares": n_ejemplares,
                    "Matriz": tipo_matriz,
                    "Modelo": res["modelo"],
                    "k": res["k"] if res["modelo"] != "dbscan" else None,
                    "best_eps": res["eps"] if res["modelo"] == "dbscan" else None,
                    "Silhouette": res["silhouette"],
                    "Calinski": res["calinski"],
                    "Davies": res["davies_bouldin"]
                })

    df_resumen = pd.DataFrame(resumen_filas)
    
    # Asegurar orden de columnas
    cols = ["Instancia", "Nº de ejemplares", "Matriz", "Modelo", "k", "best_eps", "Silhouette", "Calinski", "Davies"]
    df_resumen = df_resumen[cols]
    
    csv_path = out_dir / "resultado_resumen_final.csv"
    excel_path = out_dir / "resultado_resumen_final.xlsx"
    
    df_resumen.to_csv(csv_path, index=False, sep=";")
    df_resumen.to_excel(excel_path, index=False)
    
    print(f"Resumen generado exitosamente en {out_dir}")
    return df_resumen, csv_path

if __name__ == "__main__":
    df, path = ejecutar_experimento()
    print(df.head(10).to_string(index=False))