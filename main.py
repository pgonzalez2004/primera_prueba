from pathlib import Path

from carga_datos.data_loader import (
    cargar_configuracion,
    cargar_ejemplares,
    cargar_ubicaciones,
    cargar_registros,
    filtrar_ejemplares_por_grupo,
    filtrar_ubicaciones_comer_beber,
    filtrar_registros,
    obtener_intervalo_mes,
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'config.txt'


def main() -> None:
    config = cargar_configuracion(CONFIG_PATH)

    ejemplares = cargar_ejemplares(config)
    ubicaciones = cargar_ubicaciones(config)
    registros = cargar_registros(config)

    grupo_objetivo = config.get("grupo_ejemplar", "Enganche")
    mes_objetivo = config.get("mes", "02/2026")
    fecha_inicio, fecha_fin = obtener_intervalo_mes(mes_objetivo)

    ejemplares_enganche = filtrar_ejemplares_por_grupo(ejemplares, grupo_objetivo)
    ubicaciones_comer_beber = filtrar_ubicaciones_comer_beber(ubicaciones)

    registros_filtrados = filtrar_registros(
        registros,
        ejemplares_enganche,
        ubicaciones_comer_beber
    )

    print("=== RESUMEN ===")
    print(f"Mes configurado: {mes_objetivo}")
    print(f"Intervalo aplicado: {fecha_inicio} a {fecha_fin}")
    print(f"Ejemplares totales: {len(ejemplares)}")
    print(f"Ejemplares grupo '{grupo_objetivo}': {len(ejemplares_enganche)}")
    print(f"Ubicaciones totales: {len(ubicaciones)}")
    print(f"Ubicaciones comer/beber: {len(ubicaciones_comer_beber)}")
    print(f"Registros del intervalo: {len(registros)}")
    print(f"Registros finales filtrados: {len(registros_filtrados)}")

    print(f"\n=== EJEMPLARES '{grupo_objetivo}' (primeros 3) ===")
    for ejemplar in ejemplares_enganche[:3]:
        print(ejemplar)

    print("\n=== UBICACIONES COMER/BEBER (primeras 3) ===")
    for ubicacion in ubicaciones_comer_beber[:3]:
        print(ubicacion)

    print("\n=== REGISTROS FILTRADOS (primeros 3) ===")
    for registro in registros_filtrados[:3]:
        print(registro)


if __name__ == '__main__':
    main()