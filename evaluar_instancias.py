import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

import numpy as np




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


def construir_matriz(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    ejemplares = sorted(df["ejemplar_id"].unique())
    matriz = pd.DataFrame(0.0, index=ejemplares, columns=ejemplares)

    # Convertimos entrada a bloque temporal (minutos) para ver si coinciden
    df['bloque'] = df['entrada'].dt.floor('5min')

    # Agrupamos por lugar y bloque temporal
    for (ubi, bloque), grupo in df.groupby(["ubi_id", "bloque"]):
        ids = grupo["ejemplar_id"].unique()
        for i in ids:
            for j in ids:
                if tipo == "tiempo":
                    # Sumamos tiempo si coinciden
                    matriz.loc[i, j] += 1.0 
                else:
                    matriz.loc[i, j] += 1.0
                    
    return matriz


# =========================
# Clustering y métricas
# =========================

def preparar_x(matriz: pd.DataFrame) -> pd.DataFrame:
    X = matriz.copy()
    X = X.fillna(0)
    return X


def evaluar_modelos(datos_recibidos):
    resultados = []
    
    # 1. Preparar datos y escalar
    matriz_numerica = datos_recibidos.values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(matriz_numerica)
    n = len(X_scaled)
    
    # 2. CALCULAR EPS SOBRE LA MATRIZ (0-1) PARA QUE SEA PEQUEÑO
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(matriz_numerica) # Calculamos sobre la matriz limpia
    distancias, _ = nn.kneighbors(matriz_numerica)
    eps_optimo = np.percentile(distancias[:, 1], 50) # Percentil 50 es más estable
    
    print(f"🔧 EPS automático calculado (pequeño): {eps_optimo:.4f}")
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(datos_recibidos)
    n = len(X_scaled)
    
    # ========================================
    # KMEANS (sin cambios)
    # ========================================
    kmeans_res = []
    for k in range(2, min(8, n)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        dav = davies_bouldin_score(X_scaled, labels)
        kmeans_res.append({"modelo": "kmeans", "k": k, "eps": None, "silhouette": sil, "calinski": cal, "davies": dav})
    
    mejor_kmeans = max(kmeans_res, key=lambda x: x['calinski'])
    resultados.append(mejor_kmeans)

    # ========================================
    # AGGLOMERATIVE (sin cambios)
    # ========================================
    agg_res = []
    for k in range(2, min(6, n)):
        agg_ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X_scaled)
        labels = agg_ward.labels_
        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        dav = davies_bouldin_score(X_scaled, labels)
        agg_res.append({"modelo": "agg_ward", "k": k, "eps": None, "silhouette": sil, "calinski": cal, "davies": dav})
    
    mejor_agg = max(agg_res, key=lambda x: x['calinski'])
    resultados.append(mejor_agg)

        # ========================================
    # DBSCAN - ESTRATEGIA ROBUSTA (Corregida)
    # ========================================
    db_res = []
    nn = NearestNeighbors(n_neighbors=2).fit(X_scaled)
    distancias, _ = nn.kneighbors(X_scaled)
    mediana_dist = np.median(distancias[:, 1])
    
    for eps in [mediana_dist * 0.7, mediana_dist, mediana_dist * 1.3]:
        db = DBSCAN(eps=eps, min_samples=3).fit(X_scaled)
        labels = db.labels_
        mask = labels != -1
        
        # Solo calculamos si hay al menos 2 clusters encontrados
        if sum(mask) > 0 and len(set(labels[mask])) > 1:
            n_c = len(set(labels[mask]))
            sil = silhouette_score(X_scaled[mask], labels[mask])
            cal = calinski_harabasz_score(X_scaled[mask], labels[mask])
            dav = davies_bouldin_score(X_scaled[mask], labels[mask])
            
            # Calculamos el score aquí dentro para que siempre exista
            score = (sil * 0.6) + ((cal / 1000) * 0.4) 
            
            db_res.append({
                "k": n_c, "eps": eps, "sil": sil, 
                "cal": cal, "dav": dav, "score": score
            })

    # SELECCIÓN FINAL: Ahora el 'score' ya existe en todos los elementos de db_res
    if db_res:
        mejor = max(db_res, key=lambda x: x['score'])
        resultados.append({
            "modelo": "dbscan", "k": mejor['k'], "eps": mejor['eps'], 
            "silhouette": mejor['sil'], "calinski": mejor['cal'], "davies": mejor['dav']
        })
    else:
        # Fallback si no encuentra clusters válidos
        resultados.append({"modelo": "dbscan", "k": 0, "eps": 0, "silhouette": 0, "calinski": 0, "davies": 0})
    
    return resultados


# =========================
# Experimento principal
# =========================

def ejecutar_experimento():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    filas = []
    
    for instancia in range(1, 31):
        # 2. Genera el número de caballos que quieres
        n_ej = random.randint(20, 120)
        print(f" Instancia {instancia}: Generando {n_ej} caballos...")
        
        # 3. Prepara la configuración con esos caballos
        mis_ids = list(range(1, n_ej + 1))
        cfg = Config(seed=instancia)
        cfg.ejemplar_ids = mis_ids
        
        # 4. Genera los datos usando esa configuración
        df = rows_to_df(generate_rows(cfg))
        
        # 5. Continúa con tu lógica habitual...
        n_ej_real = len(df["ejemplar_id"].unique())
        
        for tipo_matriz in ["tiempo", "coincidencia", "coincidencia_30"]:
            # 1. Construimos la matriz sin dividir
            matriz_cruda = construir_matriz(df, tipo_matriz)
            
            # 2. Normalizamos dividiendo por el valor máximo (si el máximo es > 0)
            max_val = matriz_cruda.values.max()
            matriz = matriz_cruda / max_val
            if max_val > 0:
                matriz = matriz_cruda / max_val
            else:
                matriz = matriz_cruda

            # Imprime esto aquí también para estar 100% seguros
            print(f"DEBUG: Valor máximo enviado a evaluar_modelos: {matriz.values.max():.4f}")
            
            resultados_modelo = evaluar_modelos(matriz)
            
            if resultados_modelo:
                for res in resultados_modelo:
                    filas.append({
                        "Instancia": instancia, 
                        "Nº de ejemplares": n_ej,  # ← Número real de la función
                        "Matriz": tipo_matriz, 
                        "Modelo": res["modelo"],
                        "k": res["k"], 
                        "best_eps": round(res["eps"], 4) if res["eps"] is not None else None, # <-- AÑADE EL ROUND AQUÍ
                        "Silhouette": round(res["silhouette"], 3) if res["silhouette"] is not None else None, 
                        "Calinski": round(res["calinski"], 3) if res["calinski"] is not None else None, 
                        "Davies": round(res["davies"], 3) if res["davies"] is not None else None
                    })

    df_resumen = pd.DataFrame(filas)
    
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
    print("Iniciando experimento... (esto tardará unos 30 minutos)")
    df, path = ejecutar_experimento()
    print("¡Experimento terminado!")
    print(f"Archivo guardado en: {path}")
    print("Aquí tienes un vistazo de los resultados:")
    print(df.head(50).to_string(index=False))