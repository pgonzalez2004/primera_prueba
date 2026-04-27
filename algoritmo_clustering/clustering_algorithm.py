from __future__ import annotations

from typing import Any
import pandas as pd
from sklearn.cluster import KMeans
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