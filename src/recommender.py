"""Sistema de recomendación.

1) Documentos relacionados: similitud coseno entre embeddings + coincidencia de conceptos del grafo.
2) Capacitación: recomienda cursos según competencias actuales y objetivo profesional (con explicación).
"""
from __future__ import annotations
import numpy as np

from .nlp import habilidad_de
from .ontology import AREAS, CURSOS, HABILIDADES, NIVELES


class Recomendador:
    def __init__(self, kb, repo, extractor=None):
        self.kb, self.repo = kb, repo
        self.ext = extractor or kb.extractor
        self.M_cursos = repo.emb.transform([f"{n}. {d}. {' '.join(h)}" for n, h, _, _, _, d in CURSOS.values()])
        self.ids_cursos = list(CURSOS)
        # afinidad entre habilidades: coeficiente de solapamiento de los empleados que las tienen (datos del grafo)
        self.por_hab = {h: set(g.empleado) for h, g in kb.habilidades.groupby("habilidad")}

    def afinidad(self, h1: str, h2: str) -> float:
        a, b = self.por_hab.get(h1, set()), self.por_hab.get(h2, set())
        return len(a & b) / min(len(a), len(b)) if a and b else 0.0

    # ------------------------------------------------------------------ documentos relacionados
    def _conceptos(self, did: str) -> set[str]:
        return {self.kb.G.nodes[m]["label"] for _, m, d in self.kb.G.out_edges(did, data=True)
                if d["rel"] == "menciona"}

    def relacionados(self, doc_id: str | None = None, texto: str | None = None, k: int = 5, excluir_categoria: str | None = None) -> list[dict]:
        repo = self.repo
        if doc_id is not None:
            i = repo._idx_doc[doc_id]
            v = repo.M_doc[i]
            conc = self._conceptos(doc_id)
        else:
            v = repo.emb.transform([texto])[0]
            conc = {habilidad_de(e) if e.tipo == "habilidad" else e.canonico for e in self.ext.extraer(texto) if e.tipo in ("habilidad", "tecnologia")}
        cos = repo.M_doc @ v
        out = []
        for j, did in enumerate(repo.doc_ids):
            if did == doc_id:
                continue
            d = self.kb.documento(did)
            if excluir_categoria and d["categoria"] == excluir_categoria:
                continue
            otros = self._conceptos(did)
            comp = sorted(conc & otros)
            jac = len(conc & otros) / len(conc | otros) if (conc | otros) else 0.0
            out.append(dict(doc_id=did, titulo=d["titulo"], categoria=d["categoria"], similitud_coseno=round(float(cos[j]), 3),
                            jaccard_conceptos=round(jac, 3), puntaje=round(0.7 * float(cos[j]) + 0.3 * jac, 3),
                            conceptos_compartidos=comp))
        out.sort(key=lambda r: -r["puntaje"])
        for r in out[:k]:
            r["motivo"] = (f"similitud de contenido {r['similitud_coseno']:.2f}" +
                           (f"; comparten: {', '.join(r['conceptos_compartidos'])}" if r["conceptos_compartidos"] else ""))
        return out[:k]

    # ------------------------------------------------------------------ capacitación
    def perfil_de(self, emp_id: str) -> dict:
        hs = self.kb.habilidades[self.kb.habilidades.empleado == emp_id]
        return dict(habilidades={r.habilidad: int(r.nivel) for r in hs.itertuples()},
                    completados=set(self.kb.caps[self.kb.caps.empleado == emp_id].curso))

    def objetivo_desde_texto(self, texto: str) -> list[str]:
        return list(dict.fromkeys(h for e in self.ext.extraer(texto) if (h := habilidad_de(e))))

    def capacitacion(self, habilidades: dict[str, int], objetivos: list[str], completados: set[str] | None = None, k: int = 5) -> list[dict]:
        completados = completados or set()
        areas_obj = {HABILIDADES[h][0] for h in objetivos}
        perfil_txt = " ".join(list(habilidades) + objetivos * 2)
        sim = self.M_cursos @ self.repo.emb.transform([perfil_txt])[0]
        out = []
        for j, kid in enumerate(self.ids_cursos):
            if kid in completados:
                continue
            nom, desarrolla, nivel, pre, horas, desc = CURSOS[kid]
            faltan = [p for p in pre if p not in habilidades]
            directo = [h for h in desarrolla if h in objetivos]
            if directo:
                r_obj, por_que = 1.0, f"desarrolla el objetivo: {', '.join(directo)}"
            else:
                aff, mejor_obj = max(((self.afinidad(h, o), o) for h in desarrolla for o in objetivos), default=(0.0, ""))
                if not any(HABILIDADES[h][0] in areas_obj for h in desarrolla) or faltan:
                    continue
                r_obj = 0.3 + 0.4 * min(1.0, aff)
                por_que = f"complementa su objetivo «{mejor_obj}» (misma área; {', '.join(desarrolla)} coincide con esa habilidad en {aff:.0%} de los perfiles)"
            r_pre = 1.0 if not faltan else 0.0
            nv = max((habilidades.get(h, 0) for h in desarrolla), default=0)
            if nv == 0:
                r_niv = {"básico": 1.0, "intermedio": 0.6, "avanzado": 0.3}[nivel]
            else:
                r_niv = {"básico": 0.3 if nv >= 3 else 1.0, "intermedio": 1.0 if nv in (2, 3) else 0.5, "avanzado": 1.0 if nv >= 3 else 0.5}[nivel]
            score = 0.45 * r_obj + 0.20 * r_pre + 0.20 * r_niv + 0.15 * max(0.0, float(sim[j]))
            razones = [por_que]
            if pre:
                razones.append("prerrequisitos cumplidos: " + ", ".join(pre) if not faltan else "requiere antes: " + ", ".join(faltan))
            razones.append(f"nivel {nivel}; similitud con su perfil {max(0.0, float(sim[j])):.2f}")
            out.append(dict(curso_id=kid, curso=nom, nivel=nivel, horas=horas, puntaje=round(score, 3), desarrolla=desarrolla,
                            similitud=round(float(sim[j]), 3), por_que="; ".join(razones)))
        out.sort(key=lambda r: -r["puntaje"])
        return out[:k]
