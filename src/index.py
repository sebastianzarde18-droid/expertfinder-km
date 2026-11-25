"""Repositorio vectorial: fragmentación, embeddings y búsqueda (semántica, por palabras clave e híbrida).

Embeddings:
  * por defecto, LSA (TF-IDF + SVD truncado): embeddings densos entrenados sobre el propio corpus,
    sin descargas ni claves, reproducibles en Streamlit Cloud;
  * opcional, `EF_EMBEDDINGS=st` usa sentence-transformers (modelo preentrenado multilingüe) si está instalado.
Similitud: coseno.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2

from .nlp import normalizar, tokens_contenido
from .ontology import HABILIDADES, TECNOLOGIAS

MODELO_ST = "paraphrase-multilingual-MiniLM-L12-v2"


def fragmentar(texto: str, max_palabras: int = 70) -> list[str]:
    """Divide por líneas/oraciones y las agrupa hasta ~max_palabras para conservar el contexto."""
    partes = []
    for linea in [l.strip() for l in texto.split("\n") if l.strip()]:
        if len(linea.split()) > max_palabras:  # líneas largas: dividir por oraciones
            partes += [o.strip() for o in re.split(r"(?<=[a-záéíóú0-9\)])\.\s+(?=[A-ZÁÉÍÓÚ«(])", linea) if o.strip()]
        else:
            partes.append(linea)
    frag, actual, n = [], [], 0
    for p in partes:
        w = len(p.split())
        if actual and n + w > max_palabras:
            frag.append("\n".join(actual))
            actual, n = [], 0
        actual.append(p)
        n += w
    if actual:
        frag.append("\n".join(actual))
    return frag or [texto]


@dataclass
class Fragmento:
    doc_id: str
    n: int
    texto: str
    titulo: str
    categoria: str


class Embedder:
    def __init__(self, modo: str | None = None):
        self.modo = (modo or os.environ.get("EF_EMBEDDINGS", "lsa")).lower()
        self.nombre = "LSA (TF-IDF + SVD)"
        self._st = None
        if self.modo == "st":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._st = SentenceTransformer(MODELO_ST)
                self.nombre = f"sentence-transformers · {MODELO_ST}"
            except Exception:
                self.modo = "lsa"
        self.tfidf_dense = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1)
        self.svd = None

    def fit(self, textos: list[str]):
        if self._st is not None:
            return self
        X = self.tfidf_dense.fit_transform(textos)
        k = max(2, min(64, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def transform(self, textos: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(textos, normalize_embeddings=True))
        return l2(self.svd.transform(self.tfidf_dense.transform(textos)))


def _stem(w: str) -> str:
    for suf in ("es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def expandir_consulta(consulta: str, extractor) -> str:
    """Expansión con la ontología: añade el nombre canónico, sinónimos y tecnologías de cada concepto detectado."""
    extras = []
    for e in extractor.extraer(consulta):
        if e.tipo == "habilidad":
            extras += [e.canonico] + HABILIDADES[e.canonico][1][:3] + [t for t, (h, _, _) in TECNOLOGIAS.items() if h == e.canonico]
        elif e.tipo == "tecnologia":
            h = TECNOLOGIAS[e.canonico][0]
            extras += [e.canonico, h] + TECNOLOGIAS[e.canonico][2]
    return consulta + " " + " ".join(dict.fromkeys(extras))


class Repositorio:
    """Índice sobre documentos (y sus fragmentos)."""

    def __init__(self, docs: list[dict], extractor, embedder: Embedder | None = None):
        self.extractor = extractor
        self.docs = docs
        self.emb = embedder or Embedder()
        self.frags: list[Fragmento] = []
        for d in docs:
            for i, f in enumerate(fragmentar(d["texto"])):
                self.frags.append(Fragmento(d["id"], i, f, d["titulo"], d["categoria"]))
        textos = [f"{f.titulo}. {f.texto}" for f in self.frags]
        self.emb.fit(textos)
        self.M_frag = self.emb.transform(textos)
        self.tfidf = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", ngram_range=(1, 1))
        self.X_frag = self.tfidf.fit_transform(textos)
        self.doc_ids = [d["id"] for d in docs]
        self.M_doc = self.emb.transform([d["titulo"] + ". " + d["texto"] for d in docs])
        self._tokens_frag = [set(normalizar(t).split()) for t in textos]
        self.vocab = set(self.tfidf.vocabulary_)
        self._idx_doc = {d: i for i, d in enumerate(self.doc_ids)}

    # ---------- búsqueda a nivel fragmento ----------
    def _puntajes(self, consulta: str, modo: str, expandir: bool) -> np.ndarray:
        q = expandir_consulta(consulta, self.extractor) if expandir else consulta
        if modo == "palabras":
            qt = set(tokens_contenido(consulta))
            if not qt:
                return np.zeros(len(self.frags))
            return np.array([len(qt & t) / len(qt) for t in self._tokens_frag])
        sem = self.M_frag @ self.emb.transform([q])[0]
        if modo == "semantica":
            return np.clip(sem, 0, None)
        lex = (self.X_frag @ self.tfidf.transform([q]).T).toarray().ravel()
        return 0.5 * np.clip(sem, 0, None) + 0.5 * lex

    def buscar_fragmentos(self, consulta: str, k: int = 5, modo: str = "hibrida", expandir: bool = True) -> list[dict]:
        p = self._puntajes(consulta, modo, expandir)
        orden = np.argsort(-p)[:k]
        return [dict(doc_id=self.frags[i].doc_id, titulo=self.frags[i].titulo, categoria=self.frags[i].categoria,
                     fragmento=self.frags[i].texto, puntaje=float(p[i]), n=self.frags[i].n) for i in orden if p[i] > 0]

    def buscar_documentos(self, consulta: str, k: int = 5, modo: str = "hibrida", expandir: bool = True) -> list[dict]:
        """Ranking de documentos: puntaje del mejor fragmento de cada uno."""
        p = self._puntajes(consulta, modo, expandir)
        mejor: dict[str, tuple[float, int]] = {}
        for i, f in enumerate(self.frags):
            if f.doc_id not in mejor or p[i] > mejor[f.doc_id][0]:
                mejor[f.doc_id] = (float(p[i]), i)
        orden = sorted(mejor.items(), key=lambda kv: -kv[1][0])[:k]
        return [dict(doc_id=d, titulo=self.frags[i].titulo, categoria=self.frags[i].categoria, puntaje=s,
                     fragmento=self.frags[i].texto) for d, (s, i) in orden if s > 0]

    def soporte(self, consulta: str, contexto: str) -> tuple[float, list[str]]:
        """Fracción de conceptos y términos de la consulta que aparecen en el contexto recuperado.

        Las entidades de la ontología cuentan por su nombre canónico (p. ej. «BD» ≡ «bases de datos»)."""
        ents_q = self.extractor.extraer(consulta)
        ents_c = {(e.tipo, e.canonico) for e in self.extractor.extraer(contexto)}
        cubiertos = {m for e in ents_q for m in e.mencion.split()}
        items = [(e.tipo, e.canonico) for e in ents_q]
        faltan = [e.canonico for e in ents_q if (e.tipo, e.canonico) not in ents_c]
        ctx_stems = {_stem(w) for w in normalizar(contexto).split()}
        for t in tokens_contenido(consulta):
            if t in cubiertos:
                continue
            items.append(t)
            if _stem(t) not in ctx_stems:
                faltan.append(t)
        if not items:
            return 0.0, []
        return 1 - len(faltan) / len(items), faltan

    def cobertura_terminos(self, consulta: str) -> tuple[float, list[str]]:
        """Fracción de términos de contenido de la consulta que existen en el vocabulario del repositorio."""
        toks = tokens_contenido(consulta)
        if not toks:
            return 0.0, []
        conocidos = set(self.vocab)
        for h, (_, sins, _) in HABILIDADES.items():
            conocidos |= set(normalizar(h).split()) | {w for s in sins for w in normalizar(s).split()}
        desconocidos = [t for t in toks if t not in conocidos and not any(t == c or (len(t) > 4 and c.startswith(t[:5])) for c in conocidos)]
        return 1 - len(desconocidos) / len(toks), desconocidos
