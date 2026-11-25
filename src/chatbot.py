"""Asistente conversacional con RAG (recuperación + generación fundamentada).

Flujo: pregunta -> entidades (NLP) -> recuperación híbrida de fragmentos + hechos del grafo ->
decisión de suficiencia (evita inventar) -> redacción de la respuesta con citas [Dxxx].

Generación:
  * modo «extractivo» (por defecto, sin claves): compone la respuesta únicamente con hechos del grafo y
    oraciones recuperadas, citando cada fuente;
  * modo «LLM»: si existe ANTHROPIC_API_KEY, un modelo de lenguaje redacta usando SOLO el contexto recuperado.
"""
from __future__ import annotations
import os
import re

from .nlp import normalizar, tokens_contenido
from .ontology import TECNOLOGIAS

UMBRAL_PUNTAJE = 0.20
UMBRAL_COBERTURA = 0.65
MENSAJE_SIN_INFO = ("No encontré información suficiente en la base de conocimiento para responder esa pregunta, "
                    "así que prefiero no inventar una respuesta. Puede reformularla o consultar con Recursos Humanos.")

NO_ALMACENADOS = {"salario", "sueldo", "nomina", "edad", "domicilio", "telefono", "celular", "curp", "rfc", "enfermedad", "religion"}

PROMPT_SISTEMA = """Eres el asistente de conocimiento de TalentTech Consulting. Responde en español, de forma breve y clara,
usando EXCLUSIVAMENTE el CONTEXTO proporcionado. Cita cada afirmación con la fuente entre corchetes, por ejemplo [D012].
Si el contexto no contiene la respuesta, responde exactamente: «No tengo información suficiente en la base de conocimiento.»
No uses conocimiento externo ni inventes nombres, cifras o fechas."""


class Asistente:
    def __init__(self, kb, repo, buscador, llm_modelo: str | None = None):
        self.kb, self.repo, self.buscador = kb, repo, buscador
        self.llm_modelo = llm_modelo or os.environ.get("LLM_MODEL", "claude-haiku-5-5")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    @property
    def modo(self) -> str:
        return "LLM (Anthropic) con contexto recuperado" if self.api_key else "Extractivo (sin LLM): respuesta compuesta con hechos y fragmentos citados"

    # ------------------------------------------------------------ hechos del grafo
    def _hechos(self, pregunta: str) -> list[dict]:
        kb, hechos = self.kb, []
        ents = kb.extractor.extraer(pregunta)
        qn = normalizar(pregunta)
        for e in ents:
            if e.tipo == "empleado":
                emp = kb.empleados[kb.empleados.id == e.canonico].iloc[0]
                hs = kb.habilidades_de(e.canonico)
                lista = ", ".join(f"{r.habilidad} ({r.estado}" + (f", nivel {r.nivel}/5" if r.nivel == r.nivel and r.nivel else "") + ")" for r in hs.itertuples())
                certs = [ev["texto"] for h in hs.habilidad for ev in kb.evidencias(e.canonico, h) if ev["tipo"] == "certificación"]
                proys = [kb.nombre_proy[p] for p in kb.partic[kb.partic.empleado == e.canonico].proyecto]
                incs = list(kb.incidentes[kb.incidentes.responsable == e.canonico].id)
                txt = f"{emp.nombre} es {emp.puesto} en {emp.departamento} (disponibilidad: {emp.disponibilidad}). Habilidades: {lista or 'sin registro'}."
                if certs:
                    txt += f" Certificaciones: {'; '.join(dict.fromkeys(certs))}."
                if proys:
                    txt += f" Proyectos: {', '.join(proys)}."
                if incs:
                    txt += f" Incidentes resueltos: {', '.join(incs)}."
                doc = kb.doc_de(empleado=e.canonico, categoria="CV o perfil profesional")
                hechos.append(dict(texto=txt, fuente=doc or "grafo", tipo="empleado"))
            elif e.tipo == "incidente":
                r = kb.incidentes[kb.incidentes.id == e.canonico]
                if len(r):
                    i = r.iloc[0]
                    hechos.append(dict(texto=f"{i.id} ({i.severidad}) en {kb.nombre_proy[i.proyecto]} con {i.tecnologia}: síntoma «{i.sintoma}»; causa: {i.causa}; "
                                             f"solución: {i.solucion}; resuelto por {kb.nombre_emp[i.responsable]} en {i.horas} h.",
                                       fuente=kb.doc_de(incidente=i.id) or "grafo", tipo="incidente"))
            elif e.tipo == "proyecto":
                p = kb.proyectos[kb.proyectos.id == e.canonico].iloc[0]
                eq = [f"{kb.nombre_emp[r.empleado]} ({r.rol})" for r in kb.partic[kb.partic.proyecto == p.id].itertuples()]
                hechos.append(dict(texto=f"Proyecto {p.nombre} para {p.cliente} ({p.estado}); tecnologías: {p.tecnologias.replace('|', ', ')}; equipo: {'; '.join(eq)}.",
                                   fuente=kb.doc_de(proyecto=p.id, categoria="Informe de cierre de proyecto") or "grafo", tipo="proyecto"))
        pide_incidente = any(w in qn for w in ("incidente", "incidentes", "ticket", "falla", "caida", "error", "resolvio", "solucion", "resolver"))
        pide_quien = bool(re.search(r"\b(quien|quienes|especialista|especialistas|experto|expertos|perfil|persona|candidato)\b", qn))
        habs = [e for e in ents if e.tipo in ("habilidad", "tecnologia")]
        if habs and pide_incidente and not any(h["tipo"] == "incidente" for h in hechos):
            for e in habs[:1]:
                base = e.canonico
                for i in self.buscador.incidentes_de(base)[:4]:
                    hechos.append(dict(texto=f"{i['id']} ({i['tecnologia']}, severidad {i['severidad']}): causa {i['causa']}; solución: {i['solucion']}; "
                                             f"resuelto por {i['responsable']} en {i['horas']} h.", fuente=i["doc"] or "grafo", tipo="incidente"))
        elif habs and pide_quien and not any(h["tipo"] == "empleado" for h in hechos):
            res = self.buscador.buscar(pregunta_a_requisito(pregunta, habs), k=3)
            for r in res["resultados"]:
                hechos.append(dict(texto=f"{r['nombre']} ({r['puesto']}), cobertura {r['cobertura']}, puntaje {r['puntaje']}: {r['explicacion']}",
                                   fuente=self.kb.doc_de(empleado=r["empleado"], categoria="CV o perfil profesional") or "grafo", tipo="experto"))
        return hechos

    # ------------------------------------------------------------ respuesta
    def responder(self, pregunta: str, k: int = 4) -> dict:
        if NO_ALMACENADOS & set(normalizar(pregunta).split()):
            pol = next((d for d in self.kb.docs if d["titulo"].startswith("Política de privacidad")), None)
            cita = f" [{pol['id']}]" if pol else ""
            return dict(pregunta=pregunta, hechos=[], fragmentos=[], mejor_puntaje=0.0, cobertura_terminos=0.0, terminos_desconocidos=[],
                        modo=self.modo, suficiente=False, fundamentada=bool(pol), fuentes=[pol["id"]] if pol else [],
                        respuesta="Ese dato personal no se almacena en el directorio, por lo que no puedo responderlo. "
                                  "ExpertFinder solo guarda información profesional y la protege según la política de privacidad" + cita + ".")
        hechos = self._hechos(pregunta)
        frags = self.repo.buscar_fragmentos(pregunta, k=k, modo="hibrida", expandir=True)
        mejor = frags[0]["puntaje"] if frags else 0.0
        contexto = " ".join(f["titulo"] + " " + f["fragmento"] for f in frags[:2])
        cobertura, desconocidos = self.repo.soporte(pregunta, contexto) if frags else (0.0, [])
        suficiente = bool(hechos) or (mejor >= UMBRAL_PUNTAJE and cobertura >= UMBRAL_COBERTURA)
        base = dict(pregunta=pregunta, hechos=hechos, fragmentos=frags, mejor_puntaje=round(mejor, 3),
                    cobertura_terminos=round(cobertura, 2), terminos_desconocidos=desconocidos, modo=self.modo)
        if not suficiente:
            return dict(base, respuesta=MENSAJE_SIN_INFO, fuentes=[], fundamentada=False, suficiente=False)
        respuesta, fuentes, usado_llm = None, [], False
        if self.api_key:
            try:
                respuesta = self._llm(pregunta, hechos, frags)
                usado_llm = True
            except Exception as ex:  # sin red / sin cuota: se regresa al modo extractivo
                base["aviso"] = f"No se pudo usar el LLM ({type(ex).__name__}); se usó el modo extractivo."
        if respuesta is None:
            respuesta = self._extractiva(pregunta, hechos, frags)
        fuentes = sorted(set(re.findall(r"\[(D\d{3})\]", respuesta)))
        return dict(base, respuesta=respuesta, fuentes=fuentes, fundamentada=bool(fuentes), suficiente=True, usado_llm=usado_llm)

    def _extractiva(self, pregunta, hechos, frags) -> str:
        partes = []
        for h in hechos[:4]:
            cita = f" [{h['fuente']}]" if re.fullmatch(r"D\d{3}", h["fuente"]) else ""
            partes.append(h["texto"] + cita)
        qt = set(tokens_contenido(pregunta))
        usados, vistos = 0, set()
        for f in frags:
            if f["doc_id"] in vistos:
                continue
            vistos.add(f["doc_id"])
            if usados >= (1 if hechos else 2):
                break
            doc = self.kb.documento(f["doc_id"])
            if doc["categoria"] == "Manual o política interna":
                lineas = [l for l in doc["texto"].split("\n") if l.strip() and not l.isupper()]
                cuerpo = "\n".join(f"- {l}" for l in lineas)
                partes.append(f"Según «{doc['titulo']}» [{doc['id']}]:\n{cuerpo}")
                usados += 1
                continue
            oraciones = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", f["fragmento"]) if len(s.split()) >= 4]
            puntuadas = sorted(oraciones, key=lambda s: -len(qt & set(tokens_contenido(s))))
            mejores = [s for s in puntuadas[:2] if qt & set(tokens_contenido(s))]
            if mejores and not hechos:
                partes.append(f"En «{doc['titulo']}» [{doc['id']}]: " + " ".join(mejores))
                usados += 1
        if not partes:
            return MENSAJE_SIN_INFO
        return "\n\n".join(partes)

    def _llm(self, pregunta, hechos, frags) -> str:
        import anthropic
        ctx = []
        for h in hechos[:6]:
            ctx.append(f"[{h['fuente']}] (grafo) {h['texto']}")
        for f in frags:
            ctx.append(f"[{f['doc_id']}] {f['titulo']}: {f['fragmento']}")
        cli = anthropic.Anthropic(api_key=self.api_key)
        r = cli.messages.create(model=self.llm_modelo, max_tokens=500, system=PROMPT_SISTEMA, temperature=0,
                                messages=[{"role": "user", "content": "CONTEXTO:\n" + "\n\n".join(ctx) + f"\n\nPREGUNTA: {pregunta}"}])
        return r.content[0].text.strip()


def pregunta_a_requisito(pregunta, habs) -> str:
    """Convierte la lista de entidades detectadas en una consulta para el buscador de especialistas."""
    return " y ".join(dict.fromkeys(e.canonico for e in habs))
