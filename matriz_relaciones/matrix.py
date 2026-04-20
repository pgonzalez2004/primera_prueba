from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any

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


def calcular_matriz_coexistencia(
    registros: list[dict[str, Any]],
    ejemplares: list[dict[str, Any]],
) -> dict[int, dict[int, int]]:
    matriz = defaultdict(lambda: defaultdict(int))

    registros_norm = normalizar_registros_para_matriz(registros)
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