from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urljoin

import requests

def construir_url(config: dict[str, Any], endpoint_key: str) -> str:
    base_url = config.get("base_url")
    endpoint = config.get(endpoint_key)

    if not base_url:
        raise ValueError("Falta 'base_url' en config.txt")
    if not endpoint:
        raise ValueError(f"Falta '{endpoint_key}' en config.txt")

    return urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))

def cargar_configuracion(ruta_config):
    config: dict[str, Any] = {}
    with open(ruta_config, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            if "=" in linea:
                clave, valor = linea.split("=", 1)
                config[clave.strip()] = valor.strip()
    return config


def obtener_intervalo_mes(mes_anyo: str):
    # Formato esperado "MM/YYYY", por ejemplo "02/2026"
    mes, anyo = mes_anyo.split("/")
    mes = int(mes)
    anyo = int(anyo)

    fecha_inicio = datetime(anyo, mes, 1).date()

    if mes == 12:
        fecha_fin = datetime(anyo, 12, 31).date()
    else:
        fecha_siguiente_mes = datetime(anyo, mes + 1, 1).date()
        fecha_fin = fecha_siguiente_mes.replace(day=1) - timedelta(days=1)

    return fecha_inicio, fecha_fin


def _get_headers(config: dict[str, Any]) -> dict[str, str]:
    token = config.get("token")
    if not token:
        raise ValueError("No se encontró 'token' en config.txt")
    return {"Authorization": f"Bearer {token}"}


def cargar_ejemplares(config: dict[str, Any]) -> list[dict[str, Any]]:
    url = construir_url(config, "endpoint_ejemplares")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def cargar_ubicaciones(config: dict[str, Any]) -> list[dict[str, Any]]:
    url = construir_url(config, "endpoint_ubicaciones")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def cargar_registros(config: dict[str, Any]) -> list[dict[str, Any]]:
    url = construir_url(config, "endpoint_registros")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def filtrar_ejemplares_por_grupo(
    ejemplares: list[dict[str, Any]],
    grupo_objetivo: str,
) -> list[dict[str, Any]]:
    return [
        e for e in ejemplares
        if grupo_objetivo in (e.get("grupos") or "")
    ]


def filtrar_ubicaciones_comer_beber(
    ubicaciones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    acciones_objetivo = {"Comer", "Beber"}
    return [
        u for u in ubicaciones
        if u.get("accion") in acciones_objetivo
    ]


def filtrar_registros(
    registros: list[dict[str, Any]],
    ejemplares: list[dict[str, Any]],
    ubicaciones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ids_ejemplares = {e["id"] for e in ejemplares}
    ids_ubicaciones = {u["id"] for u in ubicaciones}

    return [
        r for r in registros
        if r.get("ejemplar") in ids_ejemplares and r.get("ubi") in ids_ubicaciones
    ]


def cargar_datos_filtrados(config: dict[str, Any]) -> dict[str, Any]:
    ejemplares = cargar_ejemplares(config)
    ubicaciones = cargar_ubicaciones(config)
    registros = cargar_registros(config)

    grupo_objetivo = config.get("grupo_ejemplar", "Enganche")
    mes_objetivo = config.get("mes", "02/2026")
    fecha_inicio, fecha_fin = obtener_intervalo_mes(mes_objetivo)

    ejemplares_enganche = filtrar_ejemplares_por_grupo(
        ejemplares,
        grupo_objetivo,
    )
    ubicaciones_comer_beber = filtrar_ubicaciones_comer_beber(ubicaciones)

    registros_filtrados = filtrar_registros(
        registros,
        ejemplares_enganche,
        ubicaciones_comer_beber,
    )

    return {
        "config": config,
        "grupo_objetivo": grupo_objetivo,
        "mes_objetivo": mes_objetivo,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "ejemplares": ejemplares,
        "ubicaciones": ubicaciones,
        "registros": registros,
        "ejemplares_enganche": ejemplares_enganche,
        "ubicaciones_comer_beber": ubicaciones_comer_beber,
        "registros_filtrados": registros_filtrados,
    }