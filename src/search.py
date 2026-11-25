"""Búsqueda de especialistas con ranking explicable.

Interpreta el requisito (lenguaje natural), lo normaliza con la ontología, recorre el grafo y calcula
un puntaje ponderado por evidencia. Cada resultado incluye el porqué de su relevancia.
"""
from __future__ import annotations
import difflib
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .index import _stem
from .nlp import habilidad_de, normalizar, tokens_contenido
from .ontology import HABILIDADES, TECNOLOGIAS

PESOS_DEFECTO = dict(declarada=0.15, nivel=0.10, certificacion=0.30, cert_extra=0.05, proyecto=0.12, proyecto_max=0.24,
                     incidente=0.12, incidente_max=0.24, capacitacion=0.08, inferida=0.10)


@dataclass
class Requisito:
    habilidades: list[str]
    tecnologias: list[str] = field(default_factory=list)
    interpretacion: str = ""
    no_reconocido: list[str] = field(default_factory=list)
    metodo: str = "exacta"  # exacta | aproximada | semantica | ninguna


class BuscadorExpertos:
    def __init__(self, kb, repo=None, pesos: dict | None = None):
        self.kb = kb
        self.repo = repo
        self.pesos = {**PESOS_DEFECTO, **(pesos or {})}
        self._lex_simple = sorted({p for p, *_ in kb.extractor.lexico if " " not in p and len(p) >= 5})
        self._vocab_stem = {_stem(w) for w in (repo.vocab if repo is not None else [])}

    # ---- interpretación de la consulta ----
    def interpretar(self, consulta: str) -> Requisito:
        ents = self.kb.extractor.extraer(consulta)
        habs, techs = [], []
        for e in ents:
            h = habilidad_de(e)
            if h and h not in habs:
                habs.append(h)
            if e.tipo == "tecnologia" and e.canonico not in techs:
                techs.append(e.canonico)
        interp = "Coincidencia con la ontología (sinónimos y tecnologías normalizados)."
        metodo, no_rec = "exacta", []
        if not habs:  # 1) tolerancia a errores de escritura; 2) similitud semántica con las descripciones de habilidades
            for tok in normalizar(consulta).split():
                for cerca in difflib.get_close_matches(tok, self._lex_simple, n=1, cutoff=0.84):
                    for e in self.kb.extractor.extraer(cerca):
                        h = habilidad_de(e)
                        if h and h not in habs:
                            habs.append(h)
            if habs:
                interp, metodo = "Corrección ortográfica aproximada contra la ontología.", "aproximada"
        if not habs and self.repo is not None:
            # La interpretación semántica solo se intenta si la consulta contiene términos que existen en el repositorio
            conocidos = [t for t in tokens_contenido(consulta) if _stem(t) in self._vocab_stem]
            if conocidos:
                desc = [f"{h}. {HABILIDADES[h][2]}. {' '.join(HABILIDADES[h][1])}" for h in HABILIDADES]
                sims = self.repo.emb.transform(desc) @ self.repo.emb.transform([" ".join(conocidos)])[0]
                orden = np.argsort(-sims)
                if sims[orden[0]] >= 0.30:
                    cand = [i for i in orden[:2] if sims[i] >= 0.85 * sims[orden[0]] and sims[i] >= 0.30]
                    habs = [list(HABILIDADES)[i] for i in cand]
                    terminos = " ".join(conocidos)
                    parecidas = " y ".join(f"{h} (coseno {sims[i]:.2f})" for h, i in zip(habs, cand))
                    prefijo = "Interpretación semántica ambigua" if len(habs) > 1 else "Interpretación semántica"
                    interp = f"{prefijo}: «{terminos}» se parece a {parecidas}. Precise la consulta si no es lo que buscaba."
                    metodo = "semantica"
        if not habs:
            no_rec = [t for t in tokens_contenido(consulta)]
            interp = "No se reconoció ninguna habilidad ni tecnología del catálogo; no se devuelven resultados para no inventar coincidencias."
            metodo = "ninguna"
        return Requisito(habs, techs, interp, no_rec, metodo)

    # ---- puntaje ----
    def _puntaje_habilidad(self, emp_id: str, h: str, techs_pref: list[str]):
        w = self.pesos
        kb = self.kb
        hr = kb._i_hab.get((emp_id, h))
        hs = [hr] if hr is not None else []
        ev = kb.evidencias(emp_id, h)
        s = 0.0
        partes = []
        if len(hs):
            nivel = int(hs[0].nivel)
            s += w["declarada"] + w["nivel"] * nivel / 5
            partes.append(f"nivel declarado {nivel}/5 ({int(hs[0].anios)} años)")
        certs = [e for e in ev if e["tipo"] == "certificación" and e["vigente"]]
        vencidas = [e for e in ev if e["tipo"] == "certificación" and not e["vigente"]]
        if certs:
            s += w["certificacion"] + min(2, len(certs) - 1) * w["cert_extra"]
        pr = [e for e in ev if e["tipo"] == "proyecto"]
        s += min(w["proyecto_max"], w["proyecto"] * len(pr))
        inc = [e for e in ev if e["tipo"] == "incidente"]
        s += min(w["incidente_max"], w["incidente"] * len(inc))
        cap = [e for e in ev if e["tipo"] == "capacitación"]
        if cap:
            s += w["capacitacion"]
        estado = kb.estado_habilidad(emp_id, h)
        if estado == "sin evidencia":
            return 0.0, estado, [], vencidas, partes
        if not len(hs):
            s = s * 0.8 + w["inferida"]  # R5: experiencia probable, pendiente de confirmación
        # bonificación si se pidió una tecnología específica y hay evidencia con ella
        bonus = 0.0
        for t in techs_pref:
            if any(t.lower() in e["texto"].lower() for e in ev):
                bonus = 0.10
        s = min(1.0, s + bonus)
        ev_validas = [e for e in ev if e["vigente"]] + []
        return s, estado, ev_validas, vencidas, partes

    def buscar(self, consulta: str, k: int = 10, solo_disponibles: bool = False, departamento: str | None = None,
               nivel_min: int = 0) -> dict:
        t0 = time.perf_counter()
        req = self.interpretar(consulta)
        kb = self.kb
        resultados = []
        for e in kb.empleados.itertuples():
            if solo_disponibles and e.disponibilidad == "ocupado":
                continue
            if departamento and e.departamento != departamento:
                continue
            detalle, total, cubiertas = {}, 0.0, 0
            for h in req.habilidades:
                s, estado, ev, venc, partes = self._puntaje_habilidad(e.id, h, req.tecnologias)
                if nivel_min:
                    hr = kb._i_hab.get((e.id, h))
                    if hr is None or int(hr.nivel) < nivel_min:
                        s = 0.0
                if s > 0:
                    cubiertas += 1
                detalle[h] = dict(puntaje=round(s, 3), estado=estado, evidencias=ev, vencidas=venc, nivel_texto=partes)
                total += s
            if not req.habilidades or cubiertas == 0:
                continue
            score = total / len(req.habilidades)
            resultados.append(dict(empleado=e.id, nombre=e.nombre, puesto=e.puesto, departamento=e.departamento,
                                   disponibilidad=e.disponibilidad, puntaje=round(100 * score, 1),
                                   cobertura=f"{cubiertas} de {len(req.habilidades)}", cubiertas=cubiertas,
                                   detalle=detalle, explicacion=self._explicar(e.nombre, detalle, cubiertas, len(req.habilidades))))
        resultados.sort(key=lambda r: (-r["puntaje"], r["nombre"]))
        return dict(consulta=consulta, requisito=req, resultados=resultados[:k], total=len(resultados),
                    tiempo_ms=round(1000 * (time.perf_counter() - t0), 1))

    @staticmethod
    def _explicar(nombre, detalle, cubiertas, n):
        frases = [f"Cumple {cubiertas} de {n} requisitos."]
        for h, d in detalle.items():
            if d["puntaje"] <= 0:
                frases.append(f"{h}: sin evidencia registrada.")
                continue
            ev = "; ".join(e["texto"] for e in d["evidencias"][:3]) or "solo declarada, sin evidencia"
            frases.append(f"{h} [{d['estado']}]: {ev}.")
            if d["vencidas"]:
                frases.append(f"  (Certificación vencida no suma: {d['vencidas'][0]['texto']}.)")
        return " ".join(frases)

    def incidentes_de(self, tecnologia_o_habilidad: str) -> list[dict]:
        """Incidentes sobre una tecnología (o sobre las tecnologías de una habilidad) y quién los resolvió."""
        kb = self.kb
        techs = [tecnologia_o_habilidad] if tecnologia_o_habilidad in TECNOLOGIAS else kb.tech_de_habilidad(tecnologia_o_habilidad)
        out = []
        for i in kb.incidentes[kb.incidentes.tecnologia.isin(techs)].itertuples():
            out.append(dict(id=i.id, tecnologia=i.tecnologia, severidad=i.severidad, causa=i.causa, solucion=i.solucion,
                            horas=int(i.horas), responsable=kb.nombre_emp[i.responsable], proyecto=kb.nombre_proy[i.proyecto],
                            doc=kb.doc_de(incidente=i.id)))
        return out
