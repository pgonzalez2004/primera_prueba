import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from matriz_relaciones import matrix

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import friedmanchisquare, wilcoxon

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
    registros = []

    for _, row in df.iterrows():
        registros.append({
            "entrada": row["entrada"].isoformat(),
            "salida": row["salida"].isoformat(),
            "ejemplar": row["ejemplar_id"],
            "ubi": row["ubi_id"],
        })

    ejemplares_ids = sorted(df["ejemplar_id"].unique())

    ejemplares = [{"id": e} for e in ejemplares_ids]

    if tipo == "tiempo":
        matriz_dict = matrix.calcular_matriz_coexistencia(registros, ejemplares)

    elif tipo == "coincidencia":
        matriz_dict = matrix.calcular_matriz_coincidencias_puras(registros)

    elif tipo == "coincidencia_30":
        matriz_dict = matrix.calcular_matriz_coincidencias_umbral(registros, 30)

    else:
        raise ValueError(f"Tipo de matriz no reconocido: {tipo}")

    matriz_df = pd.DataFrame(0.0, index=ejemplares_ids, columns=ejemplares_ids)

    for ej_a, relaciones in matriz_dict.items():
        for ej_b, valor in relaciones.items():
            matriz_df.loc[ej_a, ej_b] = valor

    return matriz_df


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
    nn = NearestNeighbors(n_neighbors=2).fit(matriz_numerica) 
    distancias, _ = nn.kneighbors(matriz_numerica)
    eps_optimo = np.percentile(distancias[:, 1], 50) 
    
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
        kmeans_res.append({
            "modelo": "kmeans",
            "k": k,
            "eps": None,
            "silhouette": sil,
            "calinski": cal,
            "davies": dav,
            "ruido": None,
            "porcentaje_ruido": None
        })
    
    mejor_kmeans = max(kmeans_res, key=lambda x: x['calinski'])
    resultados.append(mejor_kmeans)

    # ========================================
    # AGGLOMERATIVE (sin cambios)
    # ========================================
    agg_res = []
    for k in range(2, min(8, n)):
        agg_ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X_scaled)
        labels = agg_ward.labels_
        sil = silhouette_score(X_scaled, labels)
        cal = calinski_harabasz_score(X_scaled, labels)
        dav = davies_bouldin_score(X_scaled, labels)
        agg_res.append({
            "modelo": "agg_ward",
            "k": k,
            "eps": None,
            "silhouette": sil,
            "calinski": cal,
            "davies": dav,
            "ruido": None,
            "porcentaje_ruido": None
        })
    
    mejor_agg = max(agg_res, key=lambda x: x['calinski'])
    resultados.append(mejor_agg)


    # ========================================
    # DBSCAN - ESTRATEGIA ROBUSTA
    # ========================================
    db_res = []
    nn = NearestNeighbors(n_neighbors=2).fit(X_scaled)
    distancias, _ = nn.kneighbors(X_scaled)
    mediana_dist = np.median(distancias[:, 1])
    
    for eps in [mediana_dist * 0.7, mediana_dist, mediana_dist * 1.3]:
        db = DBSCAN(eps=eps, min_samples=3).fit(X_scaled)
        labels = db.labels_

        n_ruido = int(np.sum(labels == -1))
        porcentaje_ruido = round((n_ruido / len(labels)) * 100, 2)

        mask = labels != -1
        
        if sum(mask) > 0 and len(set(labels[mask])) > 1:
            n_c = len(set(labels[mask]))
            sil = silhouette_score(X_scaled[mask], labels[mask])
            cal = calinski_harabasz_score(X_scaled[mask], labels[mask])
            dav = davies_bouldin_score(X_scaled[mask], labels[mask])
            
            score = (sil * 0.6) + ((cal / 1000) * 0.4) 
            
            db_res.append({
                "k": n_c,
                "eps": eps,
                "sil": sil, 
                "cal": cal,
                "dav": dav,
                "score": score,
                "ruido": n_ruido,
                "porcentaje_ruido": porcentaje_ruido
            })

    if db_res:
        mejor = max(db_res, key=lambda x: x['score'])
        resultados.append({
            "modelo": "dbscan",
            "k": mejor['k'],
            "eps": mejor['eps'], 
            "silhouette": mejor['sil'],
            "calinski": mejor['cal'],
            "davies": mejor['dav'],
            "ruido": mejor["ruido"],
            "porcentaje_ruido": mejor["porcentaje_ruido"]
        })
    else:
        resultados.append({
            "modelo": "dbscan",
            "k": 0,
            "eps": 0,
            "silhouette": 0,
            "calinski": 0,
            "davies": 0,
            "ruido": n,
            "porcentaje_ruido": 100
        })
    
    return resultados


def crear_analisis_estadistico(df_resumen: pd.DataFrame):
    df = df_resumen.copy()

    # Convertimos métricas a numérico por seguridad
    df["Silhouette"] = pd.to_numeric(df["Silhouette"], errors="coerce")
    df["Calinski"] = pd.to_numeric(df["Calinski"], errors="coerce")
    df["Davies"] = pd.to_numeric(df["Davies"], errors="coerce")

    # Score comparativo por cada instancia y tipo de matriz
    # Silhouette y Calinski: cuanto más alto, mejor
    # Davies: cuanto más bajo, mejor
    scores = []

    for (instancia, matriz), grupo in df.groupby(["Instancia", "Matriz"]):
        g = grupo.copy()

        def normalizar_positivo(col):
            minimo = g[col].min()
            maximo = g[col].max()
            if maximo == minimo:
                return pd.Series([1] * len(g), index=g.index)
            return (g[col] - minimo) / (maximo - minimo)

        def normalizar_negativo(col):
            minimo = g[col].min()
            maximo = g[col].max()
            if maximo == minimo:
                return pd.Series([1] * len(g), index=g.index)
            return (maximo - g[col]) / (maximo - minimo)

        g["Silhouette_norm"] = normalizar_positivo("Silhouette")
        g["Calinski_norm"] = normalizar_positivo("Calinski")
        g["Davies_norm"] = normalizar_negativo("Davies")

        g["Score_Estadistico"] = (
            0.4 * g["Silhouette_norm"] +
            0.4 * g["Calinski_norm"] +
            0.2 * g["Davies_norm"]
        )

        scores.append(g)

    df_scores = pd.concat(scores, ignore_index=True)

    # Mejor modelo por cada instancia y matriz
    idx_mejor_matriz = df_scores.groupby(["Instancia", "Matriz"])["Score_Estadistico"].idxmax()
    df_mejor_por_matriz = df_scores.loc[idx_mejor_matriz].reset_index(drop=True)

    # Mejor modelo global por instancia
    idx_mejor_instancia = df_scores.groupby("Instancia")["Score_Estadistico"].idxmax()
    df_mejor_por_instancia = df_scores.loc[idx_mejor_instancia].reset_index(drop=True)

    # Ranking medio por modelo
    df_ranking = (
        df_scores
        .groupby("Modelo")
        .agg(
            Score_medio=("Score_Estadistico", "mean"),
            Silhouette_media=("Silhouette", "mean"),
            Calinski_media=("Calinski", "mean"),
            Davies_media=("Davies", "mean"),
            Veces_mejor=("Score_Estadistico", "count")
        )
        .reset_index()
        .sort_values("Score_medio", ascending=False)
    )

    # Tests estadísticos: Friedman y Wilcoxon
    tests = []

    pivot = df_scores.pivot_table(
        index=["Instancia", "Matriz"],
        columns="Modelo",
        values="Score_Estadistico"
    ).dropna()

    modelos = list(pivot.columns)

    if all(m in modelos for m in ["kmeans", "agg_ward", "dbscan"]):
        stat, p_value = friedmanchisquare(
            pivot["kmeans"],
            pivot["agg_ward"],
            pivot["dbscan"]
        )

        tests.append({
            "Test": "Friedman",
            "Comparacion": "kmeans vs agg_ward vs dbscan",
            "Estadistico": round(stat, 4),
            "p_value": round(p_value, 4),
            "Interpretacion": "Hay diferencias significativas" if p_value < 0.05 else "No hay diferencias significativas"
        })

        comparaciones = [
            ("kmeans", "agg_ward"),
            ("kmeans", "dbscan"),
            ("agg_ward", "dbscan")
        ]

        for modelo_a, modelo_b in comparaciones:
            try:
                stat_w, p_w = wilcoxon(pivot[modelo_a], pivot[modelo_b])

                tests.append({
                    "Test": "Wilcoxon",
                    "Comparacion": f"{modelo_a} vs {modelo_b}",
                    "Estadistico": round(stat_w, 4),
                    "p_value": round(p_w, 4),
                    "Interpretacion": "Hay diferencias significativas" if p_w < 0.05 else "No hay diferencias significativas"
                })

            except ValueError:
                tests.append({
                    "Test": "Wilcoxon",
                    "Comparacion": f"{modelo_a} vs {modelo_b}",
                    "Estadistico": None,
                    "p_value": None,
                    "Interpretacion": "No se puede calcular porque las diferencias son nulas o insuficientes"
                })

    df_tests = pd.DataFrame(tests)

    return df_scores, df_mejor_por_matriz, df_mejor_por_instancia, df_ranking, df_tests

# =========================
# Experimento principal
# =========================

def ejecutar_experimento():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    filas = []
    
    for instancia in range(1, 101):
        # 2. Genera el número de caballos que quieres
        n_ej = random.randint(10, 120)
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
                        "Nº de ejemplares": n_ej,
                        "Matriz": tipo_matriz, 
                        "Modelo": res["modelo"],
                        "k": res["k"], 
                        "best_eps": round(res["eps"], 4) if res["eps"] is not None else None,
                        "Silhouette": round(res["silhouette"], 3) if res["silhouette"] is not None else None, 
                        "Calinski": round(res["calinski"], 3) if res["calinski"] is not None else None, 
                        "Davies": round(res["davies"], 3) if res["davies"] is not None else None,
                        "Ruido_DBSCAN": res.get("ruido"),
                        "%_Ruido_DBSCAN": res.get("porcentaje_ruido")
                    })

    df_resumen = pd.DataFrame(filas)
    
    # Asegurar orden de columnas
    cols = [
        "Instancia",
        "Nº de ejemplares",
        "Matriz",
        "Modelo",
        "k",
        "best_eps",
        "Silhouette",
        "Calinski",
        "Davies",
        "Ruido_DBSCAN",
        "%_Ruido_DBSCAN"
    ]
    df_resumen = df_resumen[cols]
    
    df_scores, df_mejor_por_matriz, df_mejor_por_instancia, df_ranking, df_tests = crear_analisis_estadistico(df_resumen)

    analisis_path = out_dir / "analisis_estadistico_100.xlsx"

    with pd.ExcelWriter(analisis_path, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, sheet_name="resumen_resultados", index=False)
        df_scores.to_excel(writer, sheet_name="scores_modelos", index=False)
        df_mejor_por_matriz.to_excel(writer, sheet_name="mejor_por_matriz", index=False)
        df_mejor_por_instancia.to_excel(writer, sheet_name="mejor_por_instancia", index=False)
        df_ranking.to_excel(writer, sheet_name="ranking_modelos", index=False)
        df_tests.to_excel(writer, sheet_name="tests_estadisticos", index=False)

    print(f"Análisis estadístico guardado en: {analisis_path}")

    return df_resumen, analisis_path

if __name__ == "__main__":
    print("Iniciando experimento... (esto tardará unos 30 minutos)")
    df, path = ejecutar_experimento()
    print("¡Experimento terminado!")
    print(f"Archivo guardado en: {path}")
    print("Aquí tienes un vistazo de los resultados:")
    print(df.head(50).to_string(index=False))  