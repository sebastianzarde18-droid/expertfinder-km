"""Base de conocimiento: carga de datos, grafo de conocimiento (NetworkX) y evidencias.

La fuente de verdad son las tablas (CSV/JSON); el grafo se construye a partir de ellas y se
reconstruye cuando se confirman nuevas relaciones desde un documento cargado.
"""
from __future__ import annotations
import json
import re
from html import escape
from datetime import date
from pathlib import Path

import networkx as nx
import pandas as pd

from .nlp import Extractor, habilidad_de, fechas_certificado
from .ontology import (AREAS, CATEGORIAS_DOC, CERTIFICACIONES, CURSOS, FECHA_REFERENCIA,
                       HABILIDADES, TECNOLOGIAS)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLORES = {  # paleta apta para daltonismo (Okabe-Ito) + neutros
    "Empleado": "#0072B2", "Habilidad": "#E69F00", "Tecnología": "#F0E442", "Certificación": "#009E73",
    "Capacitación": "#56B4E9", "Proyecto": "#D55E00", "Incidente": "#CC79A7", "Cliente": "#999999",
    "Departamento": "#666666", "Documento": "#BBBBBB", "Categoría": "#000000", "Área": "#8C564B",
}


def _d(s):
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return None
    return date.fromisoformat(str(s)[:10])


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR, fecha_ref: date = FECHA_REFERENCIA):
        self.data_dir = Path(data_dir)
        self.fecha_ref = fecha_ref
        self.empleados = pd.read_csv(self.data_dir / "empleados.csv")
        self.habilidades = pd.read_csv(self.data_dir / "habilidades_empleado.csv")
        self.certs = pd.read_csv(self.data_dir / "certificaciones_empleado.csv")
        self.caps = pd.read_csv(self.data_dir / "capacitaciones_empleado.csv")
        self.proyectos = pd.read_csv(self.data_dir / "proyectos.csv")
        self.partic = pd.read_csv(self.data_dir / "participaciones.csv")
        self.incidentes = pd.read_csv(self.data_dir / "incidentes.csv")
        with open(self.data_dir / "documentos.json", encoding="utf-8") as f:
            self.docs: list[dict] = json.load(f)
        for d in self.docs:
            d.setdefault("estado", "confirmado")
        self.extractor = Extractor(self.empleados, self.proyectos, sorted(self.proyectos.cliente.unique()))
        self.nombre_emp = {}
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self.reconstruir()

    # ------------------------------------------------------------------ construcción
    def reconstruir(self):
        self._cache_doc = {}
        self._cache_ev = {}
        self.nombre_emp = dict(zip(self.empleados.id, self.empleados.nombre))
        self.nombre_proy = dict(zip(self.proyectos.id, self.proyectos.nombre))
        self._indices()
        G = nx.MultiDiGraph()

        def nodo(i, tipo, label, **p):
            if i not in G:
                G.add_node(i, tipo=tipo, label=label, **p)

        def arco(a, rel, b, **p):
            G.add_edge(a, b, rel=rel, **p)

        for a, desc in AREAS.items():
            nodo(f"AREA:{a}", "Área", a, descripcion=desc)
        for c in CATEGORIAS_DOC:
            nodo(f"CAT:{c}", "Categoría", c)
        for h, (area, sins, desc) in HABILIDADES.items():
            nodo(f"H:{h}", "Habilidad", h, sinonimos=", ".join(sins), descripcion=desc)
            arco(f"H:{h}", "pertenece_al_área", f"AREA:{area}")
        for t, (h, tipo, sins) in TECNOLOGIAS.items():
            nodo(f"T:{t}", "Tecnología", t, clase=tipo, sinonimos=", ".join(sins))
            arco(f"T:{t}", "es_un_tipo_de", f"H:{h}")
        for cid, (nom, emisor, h, vig) in CERTIFICACIONES.items():
            nodo(cid, "Certificación", nom, emisor=emisor)
            arco(cid, "avala", f"H:{h}")
        for kid, (nom, habs, nivel, pre, hrs, desc) in CURSOS.items():
            nodo(kid, "Capacitación", nom, nivel=nivel, horas=hrs)
            for h in habs:
                arco(kid, "desarrolla", f"H:{h}")
        for dep in self.empleados.departamento.unique():
            nodo(f"DEP:{dep}", "Departamento", dep)
        for e in self.empleados.itertuples():
            nodo(e.id, "Empleado", e.nombre, puesto=e.puesto, departamento=e.departamento, disponibilidad=e.disponibilidad)
            arco(e.id, "pertenece_a", f"DEP:{e.departamento}")
        for h in self.habilidades.itertuples():
            arco(h.empleado, "tiene_habilidad", f"H:{h.habilidad}", nivel=int(h.nivel), anios=int(h.anios))
        for c in self.certs.itertuples():
            arco(c.empleado, "posee", c.certificacion, obtencion=c.obtencion, vigencia=c.vigencia if isinstance(c.vigencia, str) else "")
        for k in self.caps.itertuples():
            arco(k.empleado, "completó", k.curso, fecha=k.fecha)
        for cli in self.proyectos.cliente.unique():
            nodo(f"CLI:{cli}", "Cliente", cli)
        for p in self.proyectos.itertuples():
            nodo(p.id, "Proyecto", p.nombre, cliente=p.cliente, estado=p.estado, sector=p.sector)
            arco(p.id, "solicitado_por", f"CLI:{p.cliente}")
            for t in p.tecnologias.split("|"):
                arco(p.id, "usa", f"T:{t}")
        for r in self.partic.itertuples():
            arco(r.empleado, "participó_en", r.proyecto, rol=r.rol)
        for i in self.incidentes.itertuples():
            nodo(i.id, "Incidente", i.id, severidad=i.severidad, causa=i.causa, solucion=i.solucion, horas=int(i.horas))
            arco(i.responsable, "resolvió", i.id)
            arco(i.id, "ocurrió_en", i.proyecto)
            arco(i.id, "involucra", f"T:{i.tecnologia}")
        # Documentos, categorías y menciones (salida del NLP)
        for d in self.docs:
            self._doc_a_grafo(G, d)
        self.G = G

    def _doc_a_grafo(self, G, d):
        did = d["id"]
        G.add_node(did, tipo="Documento", label=d["titulo"], categoria=d["categoria"], estado=d.get("estado", "confirmado"))
        G.add_edge(did, f"CAT:{d['categoria']}", rel="clasificado_como")
        if d.get("empleado") in self.nombre_emp:
            G.add_edge(did, d["empleado"], rel="respalda")
        if d.get("proyecto") in self.nombre_proy:
            G.add_edge(did, d["proyecto"], rel="documenta")
        if d.get("incidente") and d["incidente"] in G:
            G.add_edge(did, d["incidente"], rel="documenta")
        if d.get("certificacion") in CERTIFICACIONES:
            G.add_edge(did, d["certificacion"], rel="acredita")
        if d.get("curso") in CURSOS:
            G.add_edge(did, d["curso"], rel="acredita")
        for e in self.extractor.extraer(d["texto"]):
            if e.tipo == "habilidad":
                G.add_edge(did, f"H:{e.canonico}", rel="menciona")
            elif e.tipo == "tecnologia":
                G.add_edge(did, f"T:{e.canonico}", rel="menciona")

    # ------------------------------------------------------------------ consultas
    def vigente(self, vigencia) -> bool:
        v = _d(vigencia)
        return v is None or v >= self.fecha_ref

    def documento(self, did):
        return next((d for d in self.docs if d["id"] == did), None)

    def doc_de(self, **meta):
        clave = tuple(sorted(meta.items()))
        if clave not in self._cache_doc:
            self._cache_doc[clave] = next((d["id"] for d in self.docs if all(d.get(k) == v for k, v in meta.items())), None)
        return self._cache_doc[clave]

    def tech_de_habilidad(self, h):
        return [t for t, (hp, _, _) in TECNOLOGIAS.items() if hp == h]

    def _indices(self):
        """Índices por empleado para evitar filtrar tablas en cada consulta."""
        g = lambda df, col: {k: list(v.itertuples()) for k, v in df.groupby(col)}
        self._i_cert, self._i_part, self._i_cap = g(self.certs, "empleado"), g(self.partic, "empleado"), g(self.caps, "empleado")
        self._i_inc = g(self.incidentes, "responsable")
        self._tec_proy = {p.id: p.tecnologias.split("|") for p in self.proyectos.itertuples()}
        self._i_hab = {(r.empleado, r.habilidad): r for r in self.habilidades.itertuples()}

    def evidencias(self, emp_id: str, habilidad: str) -> list[dict]:
        """Evidencias de que un empleado domina una habilidad (certificación, proyecto, incidente, capacitación)."""
        if (emp_id, habilidad) in self._cache_ev:
            return self._cache_ev[(emp_id, habilidad)]
        ev = []
        techs = set(self.tech_de_habilidad(habilidad))
        for c in self._i_cert.get(emp_id, []):
            nom, emisor, h, _ = CERTIFICACIONES[c.certificacion]
            if h == habilidad:
                vig = c.vigencia if isinstance(c.vigencia, str) else ""
                ok = self.vigente(vig)
                ev.append(dict(tipo="certificación", texto=f"{nom} ({emisor})" + (f", vigente hasta {vig}" if vig and ok else
                               (f", VENCIDA el {vig}" if vig else ", sin vencimiento")),
                               vigente=ok, doc=self.doc_de(empleado=emp_id, certificacion=c.certificacion)))
        for p in self._i_part.get(emp_id, []):
            usadas = [t for t in self._tec_proy[p.proyecto] if t in techs]
            if usadas:
                ev.append(dict(tipo="proyecto", texto=f"Proyecto {self.nombre_proy[p.proyecto]} ({p.rol}); usa {', '.join(usadas)}",
                               vigente=True, doc=self.doc_de(proyecto=p.proyecto, categoria="Informe de cierre de proyecto")))
        for i in self._i_inc.get(emp_id, []):
            if i.tecnologia in techs:
                ev.append(dict(tipo="incidente", texto=f"Resolvió {i.id} ({i.tecnologia}): {i.causa}", vigente=True,
                               doc=self.doc_de(incidente=i.id)))
        for k in self._i_cap.get(emp_id, []):
            nom, habs, *_ = CURSOS[k.curso]
            if habilidad in habs:
                ev.append(dict(tipo="capacitación", texto=f"Curso «{nom}» concluido", vigente=True,
                               doc=self.doc_de(empleado=emp_id, curso=k.curso)))
        self._cache_ev[(emp_id, habilidad)] = ev
        return ev

    def estado_habilidad(self, emp_id: str, habilidad: str) -> str:
        """R1/R5: comprobada | declarada | inferida | sin evidencia."""
        declarada = (emp_id, habilidad) in self._i_hab
        ev = [e for e in self.evidencias(emp_id, habilidad) if e["vigente"]]
        if declarada and ev:
            return "comprobada"
        if declarada:
            return "declarada"
        # R5 (acotada): sin declaración, solo se infiere con evidencia sólida:
        # certificación vigente, capacitación, un incidente resuelto o al menos dos proyectos con la tecnología.
        tipos = [e["tipo"] for e in ev]
        if "certificación" in tipos or "capacitación" in tipos or "incidente" in tipos or tipos.count("proyecto") >= 2:
            return "inferida"
        return "sin evidencia"

    def habilidades_de(self, emp_id: str) -> pd.DataFrame:
        filas = []
        hs = set(self.habilidades[self.habilidades.empleado == emp_id].habilidad)
        for h in sorted(hs | {h for h in HABILIDADES if self.estado_habilidad(emp_id, h) == "inferida"}):
            r = self.habilidades[(self.habilidades.empleado == emp_id) & (self.habilidades.habilidad == h)]
            filas.append(dict(habilidad=h, nivel=int(r.nivel.iloc[0]) if len(r) else 0,
                              anios=int(r.anios.iloc[0]) if len(r) else 0, estado=self.estado_habilidad(emp_id, h)))
        return pd.DataFrame(filas, columns=["habilidad", "nivel", "anios", "estado"])

    def brechas(self) -> pd.DataFrame:
        """Análisis de brechas: nº de expertos con evidencia vigente por habilidad."""
        filas = []
        for h in HABILIDADES:
            exp = [e for e in self.empleados.id if self.estado_habilidad(e, h) == "comprobada"]
            filas.append(dict(habilidad=h, expertos_comprobados=len(exp),
                              expertos=", ".join(self.nombre_emp[e] for e in exp)))
        return pd.DataFrame(filas).sort_values(["expertos_comprobados", "habilidad"])

    def certificaciones_estado(self, dias: int = 90) -> pd.DataFrame:
        filas = []
        for c in self.certs.itertuples():
            vig = _d(c.vigencia)
            if vig is None:
                continue
            dif = (vig - self.fecha_ref).days
            estado = "vencida" if dif < 0 else ("por vencer" if dif <= dias else "vigente")
            filas.append(dict(empleado=self.nombre_emp[c.empleado], certificacion=CERTIFICACIONES[c.certificacion][0],
                              vigencia=vig.isoformat(), dias=dif, estado=estado))
        return pd.DataFrame(filas).sort_values("dias") if filas else pd.DataFrame(columns=["empleado", "certificacion", "vigencia", "dias", "estado"])

    def resumen(self) -> dict:
        tipos = pd.Series([d["tipo"] for _, d in self.G.nodes(data=True)]).value_counts().to_dict()
        rels = pd.Series([d["rel"] for *_, d in self.G.edges(data=True)]).value_counts().to_dict()
        return dict(nodos=self.G.number_of_nodes(), relaciones=self.G.number_of_edges(), tipos_nodo=tipos, tipos_relacion=rels)

    def subgrafo(self, centro: str | list[str], radio: int = 1, tipos: set[str] | None = None, max_nodos: int = 120) -> nx.MultiDiGraph:
        centros = [centro] if isinstance(centro, str) else list(centro)
        U = self.G.to_undirected(as_view=True)
        vistos = set(c for c in centros if c in self.G)
        frontera = set(vistos)
        for _ in range(radio):
            nuevos = set()
            for n in frontera:
                for m in U.neighbors(n):
                    if tipos and self.G.nodes[m]["tipo"] not in tipos and m not in vistos:
                        continue
                    nuevos.add(m)
            vistos |= nuevos
            frontera = nuevos
            if len(vistos) > max_nodos:
                break
        return self.G.subgraph(vistos).copy()

    def buscar_nodo(self, texto: str) -> list[str]:
        t = texto.lower()
        return [n for n, d in self.G.nodes(data=True) if t in d["label"].lower() or t == n.lower()]

    # ------------------------------------------------------------------ ingesta de documentos
    def proponer_relaciones(self, texto: str, categoria: str) -> tuple[list[dict], list[dict]]:
        """NLP -> (entidades, relaciones propuestas con estado «propuesto»). Nada se guarda aún."""
        ents = self.extractor.extraer_dicts(texto)
        por_tipo = lambda t: [e for e in ents if e["tipo"] == t]
        emps = por_tipo("empleado")
        prop = []

        def P(origen, rel, destino, **extra):
            prop.append(dict(origen=origen, relacion=rel, destino=destino, estado="propuesto", **extra))

        sujeto = emps[0]["canonico"] if emps else None
        if categoria == "Ticket de incidente":
            m = re.search(r"responsable:?\s*([^.\n]+)", texto, re.I)
            if m:
                rid = next((e["canonico"] for e in emps if e["mencion"] in self._n(m.group(1))), None)
                sujeto = rid or sujeto
        fechas = fechas_certificado(texto)
        if categoria in ("CV o perfil profesional", "Certificado o constancia") and sujeto:
            for e in ents:
                if e["tipo"] == "habilidad" and categoria.startswith("CV"):
                    P(sujeto, "tiene_habilidad", e["canonico"], tipo_destino="Habilidad")
                elif e["tipo"] == "tecnologia" and categoria.startswith("CV"):
                    P(sujeto, "experiencia_probable (R5)", TECNOLOGIAS[e["canonico"]][0], tipo_destino="Habilidad", via=e["canonico"])
                elif e["tipo"] == "certificacion":
                    P(sujeto, "posee", e["canonico"], tipo_destino="Certificación", **fechas)
                elif e["tipo"] == "curso":
                    P(sujeto, "completó", e["canonico"], tipo_destino="Capacitación")
        elif categoria == "Informe de cierre de proyecto":
            pr = por_tipo("proyecto")
            if pr:
                pid = pr[0]["canonico"]
                for e in emps:
                    P(e["canonico"], "participó_en", pid, tipo_destino="Proyecto")
                for e in por_tipo("tecnologia"):
                    P(pid, "usa", e["canonico"], tipo_destino="Tecnología")
        elif categoria == "Ticket de incidente":
            inc = por_tipo("incidente")
            if inc and sujeto:
                P(sujeto, "resolvió", inc[0]["canonico"], tipo_destino="Incidente")
            for e in por_tipo("tecnologia"):
                if inc:
                    P(inc[0]["canonico"], "involucra", e["canonico"], tipo_destino="Tecnología")
        return ents, prop

    @staticmethod
    def _n(s):
        from .nlp import normalizar
        return normalizar(s)

    def agregar_documento(self, titulo: str, texto: str, categoria: str, propuestas: list[dict] | None = None, empleado: str | None = None) -> dict:
        """Registra el documento en el repositorio y aplica las relaciones confirmadas (estado «confirmado»)."""
        nuevo = dict(id=f"D{len(self.docs) + 1:03d}", categoria=categoria, titulo=titulo, texto=texto.strip(),
                     estado="confirmado", temas=[])
        if empleado:
            nuevo["empleado"] = empleado
        aplicadas = []
        for p in propuestas or []:
            o, r, d = p["origen"], p["relacion"], p["destino"]
            if r == "tiene_habilidad" or r.startswith("experiencia_probable"):
                if not ((self.habilidades.empleado == o) & (self.habilidades.habilidad == d)).any():
                    self.habilidades.loc[len(self.habilidades)] = [o, d, 3, 1]
                    aplicadas.append(p)
            elif r == "posee":
                if not ((self.certs.empleado == o) & (self.certs.certificacion == d)).any():
                    self.certs.loc[len(self.certs)] = [o, d, p.get("obtencion", self.fecha_ref.isoformat()), p.get("vigencia", ""), f"TT-{nuevo['id']}"]
                    nuevo["certificacion"] = d
                    aplicadas.append(p)
            elif r == "completó":
                if not ((self.caps.empleado == o) & (self.caps.curso == d)).any():
                    self.caps.loc[len(self.caps)] = [o, d, self.fecha_ref.isoformat(), 100]
                    nuevo["curso"] = d
                    aplicadas.append(p)
            elif r == "participó_en":
                if not ((self.partic.empleado == o) & (self.partic.proyecto == d)).any():
                    pr = self.proyectos[self.proyectos.id == d].iloc[0]
                    self.partic.loc[len(self.partic)] = [o, d, "por confirmar", pr.inicio, pr.fin]
                    nuevo["proyecto"] = d
                    aplicadas.append(p)
            elif r == "resolvió":
                nuevo["incidente"] = d
                nuevo["empleado"] = o
                aplicadas.append(p)
        nuevo["temas"] = sorted({habilidad_de_ent for e in self.extractor.extraer(texto)
                                 if (habilidad_de_ent := habilidad_de(e))})
        self.docs.append(nuevo)
        self.reconstruir()
        nuevo["relaciones_aplicadas"] = len(aplicadas)
        return nuevo

    # ------------------------------------------------------------------ visualización
    def a_html(self, G: nx.MultiDiGraph, altura: str = "620px", resaltar: set[str] | None = None) -> str:
        from pyvis.network import Network
        net = Network(height=altura, width="100%", directed=True, bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
        for n, d in G.nodes(data=True):
            t = d["tipo"]
            tam = {"Empleado": 22, "Habilidad": 20, "Proyecto": 18, "Incidente": 16, "Documento": 10, "Categoría": 22, "Área": 22}.get(t, 14)
            if resaltar and n in resaltar:
                tam += 10
            detalle = "<br>".join(f"{k}: {escape(str(v))}" for k, v in d.items() if k not in ("tipo", "label") and v not in ("", None))
            net.add_node(n, label=escape(d["label"][:28]), title=f"<b>{escape(d['label'])}</b> ({t})<br>{detalle}", color=COLORES.get(t, "#999"),
                         size=tam, shape="dot", borderWidth=3 if resaltar and n in resaltar else 1)
        for a, b, d in G.edges(data=True):
            net.add_edge(a, b, label=d["rel"], title=d["rel"], arrows="to", color="#9aa0a6", font={"size": 9, "color": "#666"})
        net.set_options("""{"physics":{"solver":"forceAtlas2Based","forceAtlas2Based":{"gravitationalConstant":-60,"springLength":110},"stabilization":{"iterations":120}},
                           "interaction":{"hover":true,"navigationButtons":true,"tooltipDelay":80},"edges":{"smooth":{"type":"dynamic"}}}""")
        html = net.generate_html()
        ajuste = ("<script>(function(){function f(){try{network.fit({animation:false});}catch(e){}}"
                  "setTimeout(f,300);setTimeout(f,1200);window.addEventListener('resize',f);"
                  "try{new IntersectionObserver(function(e){if(e[0].isIntersecting){setTimeout(f,100);}}).observe(document.body);}catch(e){}"
                  "try{network.once('stabilizationIterationsDone',f);}catch(e){}})();</script>")
        return html.replace("</body>", ajuste + "</body>")
