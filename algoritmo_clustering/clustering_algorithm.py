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