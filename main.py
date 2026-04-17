from pathlib import Path

from carga_datos.data_loader import (
    cargar_configuracion, cargar_ejemplares
)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'config.txt'


def main() -> None:
    config = cargar_configuracion(CONFIG_PATH)
    ejemplares=cargar_ejemplares()
    
    print(ejemplares)
    print('\n=== RESUMEN DE EJECUCIÓN ===')
    


if __name__ == '__main__':
    main()
