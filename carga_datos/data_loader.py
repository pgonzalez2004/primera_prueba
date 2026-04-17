from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import calendar
import requests

ConfigDict = dict[str, str]
JSONList = list[dict[str, Any]]


def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def obtener_intervalo_mes(mes_texto: str) -> tuple[str, str]:
    mes, anio = mes_texto.split('/')
    mes = int(mes)
    anio = int(anio)

    ultimo_dia = calendar.monthrange(anio, mes)[1]

    fecha_inicio = f"{anio:04d}-{mes:02d}-01"
    fecha_fin = f"{anio:04d}-{mes:02d}-{ultimo_dia:02d}"

    return fecha_inicio, fecha_fin


def cargar_ejemplares(config: ConfigDict) -> Any:
    url = config['base_url'] + config['endpoint_ejemplares']
    return _get_json(url)


def cargar_ubicaciones(config: ConfigDict) -> Any:
    url = config['base_url'] + config['endpoint_ubicaciones']
    return _get_json(url)


def cargar_registros(config: ConfigDict) -> Any:
    url = config['base_url'] + config['endpoint_registros']
    fecha_inicio, fecha_fin = obtener_intervalo_mes(config['mes'])

    params = {
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin,
    }

    return _get_json(url, params=params)


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


def filtrar_ejemplares_por_grupo(
    ejemplares: list[dict[str, Any]],
    grupo: str
) -> list[dict[str, Any]]:
    return [
        e for e in ejemplares
        if str(e.get("grupos", "")).strip().lower() == grupo.strip().lower()
    ]


def filtrar_ubicaciones_comer_beber(
    ubicaciones: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    palabras_validas = {"comer", "beber", "comedero", "bebedero"}

    filtradas = []
    for ubicacion in ubicaciones:
        texto = " ".join(str(valor) for valor in ubicacion.values()).lower()
        if any(palabra in texto for palabra in palabras_validas):
            filtradas.append(ubicacion)

    return filtradas

def filtrar_registros(registros, ejemplares, ubicaciones):
    ids_ejemplares = {e["id"] for e in ejemplares}
    ids_ubicaciones = {u["id"] for u in ubicaciones}

    return [
        r for r in registros
        if r.get("ejemplar") in ids_ejemplares and r.get("ubi") in ids_ubicaciones
    ]