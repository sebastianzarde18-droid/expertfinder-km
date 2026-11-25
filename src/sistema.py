"""Fachada que integra todos los componentes: base de conocimiento + grafo + índice + clasificador +
buscador + recomendador + asistente. La app, el notebook y la evaluación usan esta clase."""
from __future__ import annotations
import time
from datetime import date
from pathlib import Path

from .chatbot import Asistente
from .classifier import ClasificadorDocs
from .index import Embedder, Repositorio
from .knowledge_graph import DATA_DIR, KnowledgeBase
from .ontology import FECHA_REFERENCIA
from .recommender import Recomendador
from .search import BuscadorExpertos


class Sistema:
    def __init__(self, data_dir: Path = DATA_DIR, fecha_ref: date = FECHA_REFERENCIA, pesos: dict | None = None):
        t0 = time.perf_counter()
        self.kb = KnowledgeBase(data_dir, fecha_ref)
        self.clasificador = ClasificadorDocs().entrenar([d["texto"] for d in self.kb.docs], [d["categoria"] for d in self.kb.docs])
        self.pesos = pesos
        self._armar_indices()
        self.tiempo_carga_s = round(time.perf_counter() - t0, 2)

    def _armar_indices(self):
        self.repo = Repositorio(self.kb.docs, self.kb.extractor, Embedder())
        self.buscador = BuscadorExpertos(self.kb, self.repo, self.pesos)
        self.recomendador = Recomendador(self.kb, self.repo)
        self.asistente = Asistente(self.kb, self.repo, self.buscador)

    def cambiar_fecha(self, fecha: date):
        self.kb.fecha_ref = fecha
        self.kb.reconstruir()
        self._armar_indices()

    # ---- flujo de ingesta: clasificar -> extraer -> proponer -> (confirmar) -> grafo + índices
    def analizar_documento(self, texto: str) -> dict:
        categoria, probs = self.clasificador.predecir(texto)
        entidades, propuestas = self.kb.proponer_relaciones(texto, categoria)
        return dict(categoria=categoria, probabilidades=probs, entidades=entidades, propuestas=propuestas)

    def registrar_documento(self, titulo: str, texto: str, categoria: str, propuestas: list[dict], corregida_por_usuario: bool = False) -> dict:
        if corregida_por_usuario:
            self.clasificador.corregir(texto, categoria)
        nuevo = self.kb.agregar_documento(titulo, texto, categoria, propuestas)
        self._armar_indices()
        return nuevo
