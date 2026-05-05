import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def crear_analisis_estadistico(df_resumen: pd.DataFrame):
    df = df_resumen.copy()

    df["Silhouette"] = pd.to_numeric(df["Silhouette"], errors="coerce")
    df["Calinski"] = pd.to_numeric(df["Calinski"], errors="coerce")
    df["Davies"] = pd.to_numeric(df["Davies"], errors="coerce")

    scores = []

    for (instancia, matriz), grupo in df.groupby(["Instancia", "Matriz"]):
        g = grupo.copy()

        def normalizar_positivo(col):
            minimo = g[col].min()
            maximo = g[col].max()
            if maximo == minimo:
                return pd.Series([1] * len(g), index=g.index)
            return (g[col] - minimo) / (maximo - minimo)

        def normalizar_negativo(col):
            minimo = g[col].min()
            maximo = g[col].max()
            if maximo == minimo:
                return pd.Series([1] * len(g), index=g.index)
            return (maximo - g[col]) / (maximo - minimo)

        g["Silhouette_norm"] = normalizar_positivo("Silhouette")
        g["Calinski_norm"] = normalizar_positivo("Calinski")
        g["Davies_norm"] = normalizar_negativo("Davies")

        g["Score_Estadistico"] = (
            0.4 * g["Silhouette_norm"] +
            0.4 * g["Calinski_norm"] +
            0.2 * g["Davies_norm"]
        )

        # Penalización sencilla para DBSCAN si mete mucho ruido
        if "Ruido_DBSCAN" in g.columns and "%_Ruido_DBSCAN" in g.columns:
            g.loc[g["Modelo"] == "dbscan", "Score_Estadistico"] = (
                g.loc[g["Modelo"] == "dbscan", "Score_Estadistico"] *
                (1 - g.loc[g["Modelo"] == "dbscan", "%_Ruido_DBSCAN"].fillna(0) / 100)
            )

        scores.append(g)

    df_scores = pd.concat(scores, ignore_index=True)

    idx_mejor_matriz = df_scores.groupby(["Instancia", "Matriz"])["Score_Estadistico"].idxmax()
    df_mejor_por_matriz = df_scores.loc[idx_mejor_matriz].reset_index(drop=True)

    idx_mejor_instancia = df_scores.groupby("Instancia")["Score_Estadistico"].idxmax()
    df_mejor_por_instancia = df_scores.loc[idx_mejor_instancia].reset_index(drop=True)

    df_ranking = (
        df_scores
        .groupby("Modelo")
        .agg(
            Score_medio=("Score_Estadistico", "mean"),
            Silhouette_media=("Silhouette", "mean"),
            Calinski_media=("Calinski", "mean"),
            Davies_media=("Davies", "mean"),
            Veces_evaluado=("Score_Estadistico", "count")
        )
        .reset_index()
        .sort_values("Score_medio", ascending=False)
    )

    tests = []

    pivot = df_scores.pivot_table(
        index=["Instancia", "Matriz"],
        columns="Modelo",
        values="Score_Estadistico"
    ).dropna()

    modelos = list(pivot.columns)

    if all(m in modelos for m in ["kmeans", "agg_ward", "dbscan"]):
        stat, p_value = friedmanchisquare(
            pivot["kmeans"],
            pivot["agg_ward"],
            pivot["dbscan"]
        )

        tests.append({
            "Test": "Friedman",
            "Metrica": "Score_Estadistico",
            "Comparacion": "kmeans vs agg_ward vs dbscan",
            "Estadistico": round(stat, 4),
            "p_value": round(p_value, 4),
            "Interpretacion": "Hay diferencias significativas" if p_value < 0.05 else "No hay diferencias significativas"
        })

        for modelo_a, modelo_b in [
            ("kmeans", "agg_ward"),
            ("kmeans", "dbscan"),
            ("agg_ward", "dbscan")
        ]:
            try:
                stat_w, p_w = wilcoxon(pivot[modelo_a], pivot[modelo_b])

                tests.append({
                    "Test": "Wilcoxon",
                    "Metrica": "Score_Estadistico",
                    "Comparacion": f"{modelo_a} vs {modelo_b}",
                    "Estadistico": round(stat_w, 4),
                    "p_value": round(p_w, 4),
                    "Interpretacion": "Hay diferencias significativas" if p_w < 0.05 else "No hay diferencias significativas"
                })

            except ValueError:
                tests.append({
                    "Test": "Wilcoxon",
                    "Metrica": "Score_Estadistico",
                    "Comparacion": f"{modelo_a} vs {modelo_b}",
                    "Estadistico": None,
                    "p_value": None,
                    "Interpretacion": "No se puede calcular"
                })

    df_tests = pd.DataFrame(tests)

    return df_scores, df_mejor_por_matriz, df_mejor_por_instancia, df_ranking, df_tests


if __name__ == "__main__":
    input_path = "output/analisis_estadistico.xlsx"

    df_resumen = pd.read_excel(input_path, sheet_name="resumen_resultados")

    df_scores, df_mejor_por_matriz, df_mejor_por_instancia, df_ranking, df_tests = crear_analisis_estadistico(df_resumen)

    output_path = "output/analisis_estadistico_final.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, sheet_name="resumen_resultados", index=False)
        df_scores.to_excel(writer, sheet_name="scores_modelos", index=False)
        df_mejor_por_matriz.to_excel(writer, sheet_name="mejor_por_matriz", index=False)
        df_mejor_por_instancia.to_excel(writer, sheet_name="mejor_por_instancia", index=False)
        df_ranking.to_excel(writer, sheet_name="ranking_modelos", index=False)
        df_tests.to_excel(writer, sheet_name="tests_estadisticos", index=False)

    print(f"Análisis estadístico generado en: {output_path}")