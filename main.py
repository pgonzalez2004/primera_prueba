from pathlib import Path

from carga_datos.data_loader import (
    cargar_configuracion,
    cargar_datos_filtrados,
)

# Fíjate que ahora solo importamos la función nueva que lo hace todo
from algoritmo_clustering.clustering_algorithm import (
    matriz_a_dataframe,
    comparar_todos_los_modelos
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'config.txt'

def main() -> None:
    # 1. Cargar datos
    config = cargar_configuracion(CONFIG_PATH)
    datos = cargar_datos_filtrados(config)
    
    df_matriz = matriz_a_dataframe(datos["matriz_coexistencia"])
    n_ejemplares = len(df_matriz)
    
    # 2. Imprimir el resumen de los datos cargados
    print(f"\nNúmero de ejemplares para clustering: {n_ejemplares}")
    
    print("\n=== RESUMEN ===")
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

    # 3. EVALUACIÓN MASIVA DE TODOS LOS MODELOS
    print("\n=== EVALUACIÓN MASIVA DE TODOS LOS MODELOS ===")
    
    # Creamos un nombre descriptivo para saber qué configuración de matriz se usó
    nombre_configuracion = f"{datos['metodo_matriz']}_{datos['umbral_segundos']}s"
    
    # Ejecutar la súper-función
    df_comparacion, texto_seleccion = comparar_todos_los_modelos(
        datos["matriz_coexistencia"], 
        nombre_matriz=nombre_configuracion
    )
    
    # Imprimir la conclusión automática redactada para el jefe
    print("\n" + texto_seleccion + "\n")
    
    # Ver el top 5 en pantalla
    print("=== TOP 5 CONFIGURACIONES ===")
    print(df_comparacion.head(5).to_string(index=False))

    # Guardar el CSV completo con todos los resultados ordenados
    ruta_csv_comparacion = r"C:\Users\PilarGonzálezBejaran\Desktop\HORSEDATA PILAR\comparacion_todos_modelos.csv"
    df_comparacion.to_csv(ruta_csv_comparacion, index=False, sep=";")
    
    print(f"\nSe ha guardado el ranking completo en: {ruta_csv_comparacion}")


if __name__ == '__main__':
    main()