from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def matriz_a_dataframe(matriz: dict[int, dict[int, int]]) -> pd.DataFrame:
    ejemplares = sorted(set(matriz.keys()) | {k for v in matriz.values() for k in v.keys()})

    data = []
    for ej in ejemplares:
        fila = {"ejemplar_id": ej}
        relaciones = matriz.get(ej, {})
        for otro in ejemplares:
            fila[f"rel_{otro}"] = relaciones.get(otro, 0)
        data.append(fila)

    return pd.DataFrame(data)


def ejecutar_kmeans(
    matriz: dict[int, dict[int, int]],
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[pd.DataFrame, float]:
    df = matriz_a_dataframe(matriz)

    X = df.drop(columns=["ejemplar_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )

    labels = modelo.fit_predict(X_scaled)
    df["cluster"] = labels

    score = silhouette_score(X_scaled, labels)

    return df, score

def ejecutar_agglomerative(
    matriz: dict[int, dict[int, int]],
    n_clusters: int = 3,
    linkage: str = "ward",
) -> tuple[pd.DataFrame, float]:
    df = matriz_a_dataframe(matriz)

    X = df.drop(columns=["ejemplar_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )

    labels = modelo.fit_predict(X_scaled)
    df["cluster"] = labels

    score = silhouette_score(X_scaled, labels)

    return df, score

def ejecutar_dbscan(
    matriz: dict[int, dict[int, int]],
    eps: float = 1.5,
    min_samples: int = 2,
) -> tuple[pd.DataFrame, float | None]:
    df = matriz_a_dataframe(matriz)

    X = df.drop(columns=["ejemplar_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = modelo.fit_predict(X_scaled)
    df["cluster"] = labels

    etiquetas_validas = set(labels)
    n_clusters = len(etiquetas_validas - {-1})
    n_ruido = list(labels).count(-1)

    score = None
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            score = silhouette_score(X_scaled[mask], labels[mask])

    return df, score


def probar_varios_k(
    matriz: dict[int, dict[int, int]],
    ks: list[int],
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> pd.DataFrame:
    df = matriz_a_dataframe(matriz)
    X = df.drop(columns=["ejemplar_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    resultados = []

    for k in ks:
        modelo = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = modelo.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        inercia = modelo.inertia_

        resultados.append({
            "k": k,
            "silhouette_score": sil,
            "calinski_harabasz": ch,
            "inercia": inercia,
        })

    return pd.DataFrame(resultados)

def probar_varios_k_agglomerative(
    matriz: dict[int, dict[int, int]],
    ks: list[int],
    linkage: str = "ward",
) -> pd.DataFrame:
    df = matriz_a_dataframe(matriz)
    X = df.drop(columns=["ejemplar_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    resultados = []

    for k in ks:
        modelo = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
        )

        labels = modelo.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)

        resultados.append({
            "k": k,
            "silhouette_score": sil,
            "calinski_harabasz": ch,
            "linkage": linkage,
        })

    return pd.DataFrame(resultados)

def probar_varios_dbscan(
    matriz: dict[int, dict[int, int]],
    eps_values: list[float],
    min_samples_values: list[int],
) -> pd.DataFrame:
    df = matriz_a_dataframe(matriz)
    X = df.drop(columns=["ejemplar_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    resultados = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            modelo = DBSCAN(
                eps=eps,
                min_samples=min_samples,
            )

            labels = modelo.fit_predict(X_scaled)

            n_clusters = len(set(labels) - {-1})
            n_ruido = list(labels).count(-1)

            sil = None
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    sil = silhouette_score(X_scaled[mask], labels[mask])

            resultados.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_ruido": n_ruido,
                "silhouette_score": sil,
            })

    return pd.DataFrame(resultados)


def generar_ks_hasta_n_ejemplares(
    matriz: dict[int, dict[int, int]]
) -> list[int]:
    df = matriz_a_dataframe(matriz)
    n_ejemplares = len(df)
    return list(range(2, n_ejemplares))

def comparar_todos_los_modelos(
    matriz: dict[int, dict[int, int]], 
    nombre_matriz: str = "matriz_actual"
) -> tuple[pd.DataFrame, str]:
    
    df = matriz_a_dataframe(matriz)
    X = df.drop(columns=["ejemplar_id"])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_ejemplares = len(df)
    ks_posibles = list(range(2, n_ejemplares))
    resultados = []

    # 1. Evaluar KMeans
    for k in ks_posibles:
        modelo = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = modelo.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        resultados.append({
            "matriz": nombre_matriz,
            "modelo": "KMeans",
            "parametros": f"k={k}",
            "k_final": k,
            "silhouette_score": sil,
            "calinski_harabasz": ch,
            "n_ruido": 0
        })

    # 2. Evaluar Agglomerative (probando ward, complete y average)
    for linkage in ["ward", "complete", "average"]:
        for k in ks_posibles:
            modelo = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = modelo.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            resultados.append({
                "matriz": nombre_matriz,
                "modelo": f"Agglomerative_{linkage}",
                "parametros": f"k={k}",
                "k_final": k,
                "silhouette_score": sil,
                "calinski_harabasz": ch,
                "n_ruido": 0
            })

    # 3. Evaluar DBSCAN (probando distintas distancias y muestras mínimas)
    eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    min_samples_values = [2, 3, 4]

    for eps in eps_values:
        for min_s in min_samples_values:
            modelo = DBSCAN(eps=eps, min_samples=min_s)
            labels = modelo.fit_predict(X_scaled)
            
            n_clusters = len(set(labels) - {-1})
            n_ruido = list(labels).count(-1)
            
            sil = None
            ch = None
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    sil = silhouette_score(X_scaled[mask], labels[mask])
                    ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
                    
            resultados.append({
                "matriz": nombre_matriz,
                "modelo": "DBSCAN",
                "parametros": f"eps={eps}_minS={min_s}",
                "k_final": n_clusters,
                "silhouette_score": sil,
                "calinski_harabasz": ch,
                "n_ruido": n_ruido
            })

    # Convertir a DataFrame y ordenar de mejor a peor Silhouette
    df_res = pd.DataFrame(resultados)
    df_res = df_res.sort_values(by="silhouette_score", ascending=False, na_position="last").reset_index(drop=True)

    # Redactar conclusión automática basada en el mejor resultado
    mejor = df_res.iloc[0]
    explicacion = (
        f"SELECCIÓN DEL MEJOR MODELO:\n"
        f"- Modelo ganador: {mejor['modelo']}\n"
        f"- Configuración: {mejor['parametros']}\n"
        f"- Grupos formados: {mejor['k_final']} clusters\n\n"
        f"¿Por qué se elige esta opción?\n"
        f"Porque ha obtenido el índice de Silhouette más alto ({mejor['silhouette_score']:.4f}). "
        f"Esto indica matemáticamente que es la configuración que logra que los caballos de un mismo grupo sean lo más parecidos posible entre sí en su comportamiento, y a la vez, que los grupos estén lo más separados/diferenciados posible los unos de los otros."
    )
    if "DBSCAN" in mejor['modelo']:
        explicacion += f"\nAdemás, este modelo ha detectado {mejor['n_ruido']} caballos atípicos (ruido) que no encajan en ningún grupo."

    return df_res, explicacion