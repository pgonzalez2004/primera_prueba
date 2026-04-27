from pathlib import Path

from carga_datos.data_loader import (
    cargar_configuracion,
    cargar_datos_filtrados,
)

from algoritmo_clustering.clustering_algorithm import (
    matriz_a_dataframe,
    ejecutar_kmeans,
    probar_varios_k,
    generar_ks_hasta_n_ejemplares,
    ejecutar_agglomerative,
    probar_varios_k_agglomerative,
    ejecutar_dbscan,
    probar_varios_dbscan,
)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.txt"


def main() -> None:
    config = cargar_configuracion(CONFIG_PATH)
    datos = cargar_datos_filtrados(config)

    # --- Info básica sobre la matriz ---
    df_matriz = matriz_a_dataframe(datos["matriz_coexistencia"])
    n_ejemplares = len(df_matriz)

    print(f"\nNúmero de ejemplares para clustering: {n_ejemplares}")

    # ============================================================
    # 1. KMEANS (k fijo = 3) + evaluación de varios k
    # ============================================================

    # Ejecutar KMeans con k=3
    resultado_kmeans, score_kmeans = ejecutar_kmeans(
        datos["matriz_coexistencia"],
        n_clusters=3,
    )

    # Lista de k basada en el número de ejemplares
    ks = generar_ks_hasta_n_ejemplares(datos["matriz_coexistencia"])
    print(f"Valores de k que se van a probar (KMeans): de {ks[0]} a {ks[-1]}")

    evaluacion_k = probar_varios_k(
        datos["matriz_coexistencia"],
        ks=ks,
    )

    # ============================================================
    # 2. AGGLOMERATIVE (linkage ward)
    # ============================================================

    ks = generar_ks_hasta_n_ejemplares(datos["matriz_coexistencia"])
    print(f"Valores de k que se van a probar (Agglomerative): de {ks[0]} a {ks[-1]}")

    evaluacion_agglomerative = probar_varios_k_agglomerative(
        datos["matriz_coexistencia"],
        ks=ks,
        linkage="ward",
    )

    evaluacion_agglomerative.to_csv(
        "salida_evaluacion_agglomerative.csv",
        index=False,
        sep=";",
    )

    resultado_clusters_agglom, score_agglom = ejecutar_agglomerative(
        datos["matriz_coexistencia"],
        n_clusters=3,
        linkage="ward",
    )

    resultado_clusters_agglom.to_csv(
        "salida_clusters_agglomerative_k3.csv",
        index=False,
        sep=";",
    )
    print(f"Silhouette Agglomerative (k=3): {score_agglom}")

    # ============================================================
    # 3. DBSCAN
    # ============================================================

    evaluacion_dbscan = probar_varios_dbscan(
        datos["matriz_coexistencia"],
        eps_values=[0.5, 1.0, 1.5, 2.0, 2.5],
        min_samples_values=[2, 3, 4],
    )

    evaluacion_dbscan.to_csv(
        "salida_evaluacion_dbscan.csv",
        index=False,
        sep=";",
    )

    resultado_dbscan, score_dbscan = ejecutar_dbscan(
        datos["matriz_coexistencia"],
        eps=1.5,
        min_samples=2,
    )

    resultado_dbscan.to_csv(
        "salida_clusters_dbscan.csv",
        index=False,
        sep=";",
    )
    print(f"Silhouette DBSCAN (sin ruido): {score_dbscan}")

    # ============================================================
    # 4. RESUMEN POR PANTALLA
    # ============================================================

    print("=== RESUMEN ===")
    print(f"Mes configurado: {datos['mes_objetivo']}")
    print(f"Intervalo aplicado: {datos['fecha_inicio']} a {datos['fecha_fin']}")
    print(f"Ejemplares totales: {len(datos['ejemplares'])}")
    print(f"Ejemplares grupo '{datos['grupo_objetivo']}': {len(datos['ejemplares_enganche'])}")
    print(f"Ubicaciones totales: {len(datos['ubicaciones'])}")
    print(f"Ubicaciones comer/beber: {len(datos['ubicaciones_comer_beber'])}")
    print(f"Registros del intervalo: {len(datos['registros'])}")
    print(f"Registros finales filtrados: {len(datos['registros_filtrados'])}")
    print("\n=== MATRIZ COEXISTENCIA ===")
    print(datos["matriz_coexistencia"])
    print(f"Método matriz: {datos['metodo_matriz']}")
    print(f"Umbral (segundos): {datos['umbral_segundos']}")

    print(f"\n=== EJEMPLARES '{datos['grupo_objetivo']}' (primeros 3) ===")
    for ejemplar in datos["ejemplares_enganche"][:3]:
        print(ejemplar)

    print("\n=== UBICACIONES COMER/BEBER (primeras 3) ===")
    for ubicacion in datos["ubicaciones_comer_beber"][:3]:
        print(ubicacion)

    print("\n=== REGISTROS FILTRADOS (primeros 3) ===")
    for registro in datos["registros_filtrados"][:3]:
        print(registro)

    print("\n=== CLUSTERING KMEANS ===")
    print(f"Silhouette score (k=3): {score_kmeans:.4f}")
    print(
        resultado_kmeans[["ejemplar_id", "cluster"]]
        .sort_values(["cluster", "ejemplar_id"])
        .to_string(index=False)
    )

    print("\n=== EVALUACIÓN DISTINTOS K (KMEANS) ===")
    print(evaluacion_k.to_string(index=False))

    print("\nTamaño de cada cluster (KMeans, k=3):")
    print(resultado_kmeans["cluster"].value_counts())

    # Guardar resultados KMeans
    resultado_kmeans.to_csv(
        r"C:\Users\PilarGonzálezBejaran\Desktop\HORSEDATA PILAR\salida_clusters_k3.csv",
        index=False,
        sep=";",
    )

    evaluacion_k.to_csv(
        r"C:\Users\PilarGonzálezBejaran\Desktop\HORSEDATA PILAR\salida_evaluacion_k.csv",
        index=False,
        sep=";",
    )

    print(
        "\nFicheros guardados: "
        "salida_clusters_k3.csv, salida_evaluacion_k.csv, "
        "salida_evaluacion_agglomerative.csv, salida_clusters_agglomerative_k3.csv, "
        "salida_evaluacion_dbscan.csv, salida_clusters_dbscan.csv"
    )


if __name__ == "__main__":
    main()