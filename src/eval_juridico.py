"""Prueba de generalización con documentos jurídicos de estructura y redacción variadas (enfoque académico).

Corpus: 30 textos (6 temas × 5 tipos documentales), escritos con vocabulario distinto al de las plantillas de TalentTech.
Se evalúa SIN modificar el sistema: (1) clasificación del tipo de documento (validación cruzada) y (2) recuperación por tema con
consultas literales y parafraseadas. Para usar documentos reales de Dani: ponerlos en el mismo formato JSON (id, tema, tipo, texto)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .classifier import ClasificadorDocs
from .index import Repositorio, Embedder
from .nlp import Extractor
from .evaluation import REPORTES

D = Path(__file__).resolve().parent.parent / "data_juridico"

def main():
    docs = json.load(open(D / "documentos_juridicos.json", encoding="utf-8"))
    for d in docs: d["titulo"], d["categoria"] = d["tipo"] + " " + d["id"], d["tipo"]
    X, y = [d["texto"] for d in docs], [d["tipo"] for d in docs]
    # (1) clasificación: el mismo modelo del sistema, validación cruzada de 5 pliegues
    clf = ClasificadorDocs().modelo if hasattr(ClasificadorDocs(), "modelo") else None
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", ngram_range=(1, 2)), LogisticRegression(max_iter=2000, C=10))
    pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    clasif = dict(n=len(docs), exactitud=round(accuracy_score(y, pred), 3), precision_macro=round(p, 3), recall_macro=round(r, 3), f1_macro=round(f, 3))
    # (2) recuperación: Precisión@5 y Recall@5 por tema (relevantes = los 5 documentos del tema)
    repo = Repositorio(docs, Extractor(), Embedder())
    consultas = json.load(open(D / "consultas.json", encoding="utf-8"))
    filas = []
    for c in consultas:
        rel = {d["id"] for d in docs if d["tema"] == c["tema"]}
        for modo, exp in (("palabras", False), ("semantica", False), ("hibrida", True)):
            top = [r["doc_id"] for r in repo.buscar_documentos(c["consulta"], k=5, modo=modo, expandir=exp)]
            filas.append(dict(consulta=c["consulta"], tipo=c["tipo"], metodo=modo, p5=len(set(top) & rel) / 5, r5=len(set(top) & rel) / len(rel),
                              primero_relevante=bool(top and top[0] in rel)))
    df = pd.DataFrame(filas)
    df.to_csv(REPORTES / "juridico_recuperacion_detalle.csv", index=False)
    res = df.groupby(["tipo", "metodo"])[["p5", "r5", "primero_relevante"]].mean().round(3)
    res.to_csv(REPORTES / "juridico_recuperacion_resumen.csv")
    out = dict(clasificacion=clasif, recuperacion=json.loads(res.reset_index().to_json(orient="records")))
    (REPORTES / "juridico_resumen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(clasif)); print(res.to_string())
    pd.DataFrame({"id": [d["id"] for d in docs], "real": y, "predicho": pred}).query("real != predicho").to_csv(REPORTES / "juridico_errores_clasificacion.csv", index=False)

if __name__ == "__main__":
    main()
