from pathlib import Path

from carga_datos.data_loader import (
    cargar_configuracion,
    cargar_datos_filtrados,
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'config.txt'


def main() -> None:
    config = cargar_configuracion(CONFIG_PATH)
    datos = cargar_datos_filtrados(config)

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


if __name__ == '__main__':
    main()