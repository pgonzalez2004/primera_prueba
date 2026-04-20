from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
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
    # MODO SIMULACIÓN: filtramos solo por ubicaciones de comer/beber
    ids_ubicaciones = {u["id"] for u in ubicaciones}

    return [
        r for r in registros
        if r.get("ubi") in ids_ubicaciones
    ]


def cargar_datos_filtrados(config: dict[str, Any]) -> dict[str, Any]:
    ejemplares = cargar_ejemplares(config)
    ubicaciones = cargar_ubicaciones(config)
    # registros = cargar_registros(config)
    registros = cargar_registros_ficticios("registros_sesiones_generados.csv")

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
    
    metodo_matriz = config.get("metodo_matriz", "tiempo_total")
    umbral_segundos = int(config.get("umbral_segundos", 30))

    if metodo_matriz == "tiempo_total":
        matriz_coexistencia = calcular_matriz_coexistencia(
            registros_filtrados,
            ejemplares_enganche,
        )
    elif metodo_matriz == "coincidencias":
        matriz_coexistencia = calcular_matriz_coincidencias_puras(
            registros_filtrados,
        )
    elif metodo_matriz == "coincidencias_umbral":
        matriz_coexistencia = calcular_matriz_coincidencias_umbral(
            registros_filtrados,
            umbral_segundos,
        )
    else:
        matriz_coexistencia = calcular_matriz_coexistencia(
            registros_filtrados,
            ejemplares_enganche,
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
        "matriz_coexistencia": matriz_coexistencia,
        "metodo_matriz": metodo_matriz,
        "umbral_segundos": umbral_segundos,
        "matriz_coexistencia": matriz_coexistencia,
    }
    
def parsear_fecha_iso(fecha_str: str) -> datetime:
    return datetime.fromisoformat(fecha_str)


def calcular_solape_segundos(
    entrada_a: datetime,
    salida_a: datetime,
    entrada_b: datetime,
    salida_b: datetime,
) -> int:
    inicio_solape = max(entrada_a, entrada_b)
    fin_solape = min(salida_a, salida_b)
    return max(0, int((fin_solape - inicio_solape).total_seconds()))


def normalizar_registros_para_matriz(
    registros: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    registros_normalizados = []

    for r in registros:
        entrada = r.get("entrada")
        salida = r.get("salida")
        ejemplar = r.get("ejemplar")
        ubicacion = r.get("ubi")

        if not entrada or not salida or ejemplar is None or ubicacion is None:
            continue

        registros_normalizados.append({
            "ejemplar": ejemplar,
            "ubi": ubicacion,
            "entrada": parsear_fecha_iso(entrada),
            "salida": parsear_fecha_iso(salida),
        })

    return registros_normalizados


from collections import defaultdict
from itertools import combinations

def calcular_matriz_coexistencia(
    registros: list[dict[str, Any]],
    ejemplares: list[dict[str, Any]],
) -> dict[int, dict[int, int]]:
    matriz = defaultdict(lambda: defaultdict(int))

    registros_norm = normalizar_registros_para_matriz(registros)

    # MODO SIMULACIÓN: usamos todos los ejemplares presentes en los registros
    registros_validos = registros_norm

    registros_por_ubicacion: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in registros_validos:
        registros_por_ubicacion[r["ubi"]].append(r)

    for _, regs_ubi in registros_por_ubicacion.items():
        for reg_a, reg_b in combinations(regs_ubi, 2):
            ej_a = reg_a["ejemplar"]
            ej_b = reg_b["ejemplar"]

            if ej_a == ej_b:
                continue

            solape = calcular_solape_segundos(
                reg_a["entrada"],
                reg_a["salida"],
                reg_b["entrada"],
                reg_b["salida"],
            )

            if solape > 0:
                matriz[ej_a][ej_b] += solape
                matriz[ej_b][ej_a] += solape

    return {k: dict(v) for k, v in matriz.items()}

def calcular_matriz_coincidencias_puras(
    registros: list[dict[str, Any]],
) -> dict[int, dict[int, int]]:
    """
    Matriz donde cada célula es el número de veces que dos ejemplares coinciden
    en una misma ubicación, sin importar cuántos segundos compartan.
    """
    matriz = defaultdict(lambda: defaultdict(int))
    registros_norm = normalizar_registros_para_matriz(registros)
    registros_por_ubicacion: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for r in registros_norm:
        registros_por_ubicacion[r["ubi"]].append(r)

    for _, regs_ubi in registros_por_ubicacion.items():
        for reg_a, reg_b in combinations(regs_ubi, 2):
            ej_a = reg_a["ejemplar"]
            ej_b = reg_b["ejemplar"]
            if ej_a == ej_b:
                continue

            solape = calcular_solape_segundos(
                reg_a["entrada"], reg_a["salida"],
                reg_b["entrada"], reg_b["salida"],
            )
            if solape > 0:
                matriz[ej_a][ej_b] += 1
                matriz[ej_b][ej_a] += 1

    return {k: dict(v) for k, v in matriz.items()}


def calcular_matriz_coincidencias_umbral(
    registros: list[dict[str, Any]],
    umbral_segundos: int,
) -> dict[int, dict[int, int]]:
    """
    Matriz donde cada célula es el número de veces que dos ejemplares coinciden
    en una ubicación y el tiempo compartido supera un umbral X (en segundos).
    """
    matriz = defaultdict(lambda: defaultdict(int))
    registros_norm = normalizar_registros_para_matriz(registros)
    registros_por_ubicacion: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for r in registros_norm:
        registros_por_ubicacion[r["ubi"]].append(r)

    for _, regs_ubi in registros_por_ubicacion.items():
        for reg_a, reg_b in combinations(regs_ubi, 2):
            ej_a = reg_a["ejemplar"]
            ej_b = reg_b["ejemplar"]
            if ej_a == ej_b:
                continue

            solape = calcular_solape_segundos(
                reg_a["entrada"], reg_a["salida"],
                reg_b["entrada"], reg_b["salida"],
            )
            if solape >= umbral_segundos:
                matriz[ej_a][ej_b] += 1
                matriz[ej_b][ej_a] += 1

    return {k: dict(v) for k, v in matriz.items()}

import csv
from pathlib import Path

def cargar_registros_ficticios(desde_csv: str | Path) -> list[dict[str, Any]]:
    ruta = Path(desde_csv)
    registros: list[dict[str, Any]] = []

    with ruta.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            registros.append({
                "id": int(row["id"]),
                "entrada": row["entrada"],
                "salida": row["salida"],
                "duracion": int(row["duracion"]),
                "detecciones": int(row["detecciones"]),
                "ejemplar": int(row["ejemplar_id"]),
                "ubi": int(row["ubi_id"]),
            })

    return registros