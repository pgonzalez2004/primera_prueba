from pathlib import Path

from carga_datos.data_loader import (
    cargar_configuracion,
    cargar_datos_filtrados,
)

from algoritmo_clustering.clustering_algorithm import ejecutar_kmeans, probar_varios_k, matriz_a_dataframe

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'config.txt'


def main() -> None:
    config = cargar_configuracion(CONFIG_PATH)
    datos = cargar_datos_filtrados(config)
    
    df_matriz = matriz_a_dataframe(datos["matriz_coexistencia"])
    n_ejemplares = len(df_matriz)
    ks = list(range(2, n_ejemplares))
    
    print(f"\nNúmero de ejemplares para clustering: {n_ejemplares}")
    print(f"Valores de k que se van a probar: de 2 a {n_ejemplares}")
    
    
    resultado_kmeans, score_kmeans = ejecutar_kmeans(
        datos["matriz_coexistencia"],
        n_clusters=3,
    )

    evaluacion_k = probar_varios_k(
        datos["matriz_coexistencia"],
        ks= ks,
    )

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
    for ejemplar in datos['ejemplares_enganche'][:3]:
        print(ejemplar)

    print("\n=== UBICACIONES COMER/BEBER (primeras 3) ===")
    for ubicacion in datos['ubicaciones_comer_beber'][:3]:
        print(ubicacion)

    print("\n=== REGISTROS FILTRADOS (primeros 3) ===")
    for registro in datos['registros_filtrados'][:3]:
        print(registro)

    print("\n=== CLUSTERING KMEANS ===")
    print(f"Silhouette score (k=3): {score_kmeans:.4f}")
    print(resultado_kmeans[["ejemplar_id", "cluster"]].sort_values(["cluster", "ejemplar_id"]).to_string(index=False))

    ("\n=== EVALUACIÓN DISTINTOS K ===")
    print(evaluacion_k.to_string(index=False))

    print("\nTamaño de cada cluster (k=3):")
    print(resultado_kmeans["cluster"].value_counts())
    
    resultado_kmeans.to_csv(
        r"C:\Users\PilarGonzálezBejaran\Desktop\HORSEDATA PILAR\salida_clusters_k3.csv",
        index=False,
        sep=';',
    )

    evaluacion_k.to_csv(
        r"C:\Users\PilarGonzálezBejaran\Desktop\HORSEDATA PILAR\salida_evaluacion_k.csv",
        index=False,
        sep=';',
    )
    print("\nFicheros guardados: salida_clusters_k3.csv y salida_evaluacion_k.csv")

if __name__ == '__main__':
    main()