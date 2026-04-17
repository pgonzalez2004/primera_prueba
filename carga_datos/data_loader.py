
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ConfigDict = dict[str, str]
JSONList = list[dict[str, Any]]



def cargar_configuracion(ruta_config: str | Path) -> ConfigDict:
    config: ConfigDict = {}
    ruta = Path(ruta_config)

    for linea in ruta.read_text(encoding='utf-8').splitlines():
        linea = linea.strip()
        if not linea or linea.startswith('#'):
            continue
        if '=' not in linea:
            continue
        clave, valor = linea.split('=', 1)
        config[clave.strip()] = valor.strip()

    return config
