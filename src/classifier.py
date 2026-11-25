"""Clasificación automática de documentos (NLP + Machine Learning).

Pipeline: limpieza -> TF-IDF (unigramas y bigramas) -> Regresión Logística.
Incluye comparación con otros modelos mediante validación cruzada estratificada y
reentrenamiento con correcciones manuales del usuario.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .nlp import normalizar


def limpiar(texto: str) -> str:
    """Limpieza: minúsculas, sin acentos ni signos; los códigos (INC-0457, folios) se sustituyen por marcas."""
    import re
    t = re.sub(r"INC-\d{4}", " codigoinc ", texto, flags=re.I)
    t = re.sub(r"TT-\d{4,6}", " codigofolio ", t)
    t = re.sub(r"\d{2}/\d{2}/\d{4}", " fechadoc ", t)
    return normalizar(t)


def _pipe(modelo: str = "logreg") -> Pipeline:
    vec = TfidfVectorizer(preprocessor=limpiar, sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    clf = {"logreg": LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
           "svm": LinearSVC(C=1.0, class_weight="balanced"),
           "naive_bayes": ComplementNB(alpha=0.3)}[modelo]
    return Pipeline([("tfidf", vec), ("clf", clf)])


UMBRAL_CONFIANZA = 0.50  # por debajo, el sistema pide revisión manual


class ClasificadorDocs:
    def __init__(self):
        self.pipe = _pipe("logreg")
        self.extra_x: list[str] = []
        self.extra_y: list[str] = []
        self.base_x: list[str] = []
        self.base_y: list[str] = []

    def entrenar(self, textos, etiquetas):
        self.base_x, self.base_y = list(textos), list(etiquetas)
        self.pipe = _pipe("logreg").fit(self.base_x + self.extra_x, self.base_y + self.extra_y)
        return self

    def predecir(self, texto: str) -> tuple[str, dict[str, float]]:
        p = self.pipe.predict_proba([texto])[0]
        clases = self.pipe.classes_
        orden = np.argsort(-p)
        return str(clases[orden[0]]), {str(clases[i]): float(p[i]) for i in orden}

    def corregir(self, texto: str, categoria_correcta: str):
        """Registra la corrección manual y reentrena el modelo (aprendizaje con retroalimentación)."""
        self.extra_x.append(texto)
        self.extra_y.append(categoria_correcta)
        self.pipe = _pipe("logreg").fit(self.base_x + self.extra_x, self.base_y + self.extra_y)

    # ------------------------------------------------------------ evaluación
    @staticmethod
    def evaluar(textos, etiquetas, k: int = 5, semilla: int = 0) -> dict:
        textos, etiquetas = np.array(textos, dtype=object), np.array(etiquetas)
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
        comparacion, preds = [], {}
        for nombre in ("logreg", "svm", "naive_bayes"):
            y_pred = cross_val_predict(_pipe(nombre), textos, etiquetas, cv=cv)
            rep = classification_report(etiquetas, y_pred, output_dict=True, zero_division=0)
            comparacion.append(dict(modelo=nombre, accuracy=rep["accuracy"], precision_macro=rep["macro avg"]["precision"],
                                    recall_macro=rep["macro avg"]["recall"], f1_macro=rep["macro avg"]["f1-score"]))
            preds[nombre] = y_pred
        clases = sorted(set(etiquetas))
        y_pred = preds["logreg"]
        rep = pd.DataFrame(classification_report(etiquetas, y_pred, output_dict=True, zero_division=0)).T
        cm = confusion_matrix(etiquetas, y_pred, labels=clases)
        errores = [dict(indice=int(i), real=etiquetas[i], predicha=y_pred[i]) for i in np.where(etiquetas != y_pred)[0]]
        return dict(comparacion=pd.DataFrame(comparacion), reporte=rep, matriz=cm, clases=clases, errores=errores, y_true=etiquetas, y_pred=y_pred)
