"""Diagramas del sistema generados con Python + Graphviz (arquitectura, ontología y flujo de ingesta)
y una vista estática del grafo con NetworkX/Matplotlib. Ejecutar:  python -m src.diagrams"""
from __future__ import annotations
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .knowledge_graph import COLORES
from .sistema import Sistema

FIG = Path(__file__).resolve().parent.parent / "figures"
FUENTE = "Helvetica"


def _dot(nombre: str, codigo: str, dpi: int = 170):
    import re
    FIG.mkdir(exist_ok=True)
    codigo = re.sub(r"#([0-9A-Fa-f])([0-9A-Fa-f])([0-9A-Fa-f])(?![0-9A-Fa-f])", r"#\1\1\2\2\3\3", codigo)  # graphviz exige #RRGGBB
    (FIG / f"{nombre}.dot").write_text(codigo, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", f"-Gdpi={dpi}", str(FIG / f"{nombre}.dot"), "-o", str(FIG / f"{nombre}.png")], check=True)


def _txt(t):
    return "#000000" if t in ("Tecnología", "Documento") else "#FFFFFF"


def arquitectura(S: Sistema):
    r = S.kb.resumen()
    cod = f'''digraph G {{
  rankdir=TB; compound=true; nodesep=0.3; ranksep=0.5; fontname="{FUENTE}"; bgcolor="white"; pad=0.25;
  node [shape=box, style="rounded,filled", fontname="{FUENTE}", fontsize=11, margin="0.14,0.08", color="#445"];
  edge [fontname="{FUENTE}", fontsize=9, color="#556"];

  subgraph cluster_f {{ label="1. Fuentes de datos (simuladas)"; style="rounded,filled"; fillcolor="#EEF3F8"; color="#9AB";
    csv [label="Tablas CSV\\nempleados · habilidades ·\\ncertificaciones · proyectos ·\\nincidentes · capacitaciones", fillcolor="#FFFFFF"];
    docs [label="documentos.json\\n{len(S.kb.docs)} documentos\\n(CV, certificados, informes,\\ntickets, minutas, manuales)", fillcolor="#FFFFFF"];
    onto [label="Ontología (ontology.py)\\n15 habilidades · 25 tecnologías\\n17 certificaciones · 24 cursos", fillcolor="#FFFFFF"];
  }}
  subgraph cluster_p {{ label="2. Procesamiento (Python)"; style="rounded,filled"; fillcolor="#FFF6E5"; color="#C9A";
    nlp [label="NLP: normalización y NER\\nguiado por la ontología\\n(sinónimos → nombre canónico, R6)", fillcolor="#FFFFFF"];
    clf [label="Clasificador de documentos\\nTF-IDF + Regresión Logística\\n(scikit-learn)", fillcolor="#FFFFFF"];
    emb [label="Fragmentación + embeddings\\nLSA (TF-IDF + SVD)\\nsimilitud coseno", fillcolor="#FFFFFF"];
    kgb [label="Construcción del grafo\\n(NetworkX) y reglas R1–R5", fillcolor="#FFFFFF"];
  }}
  subgraph cluster_r {{ label="3. Repositorio de conocimiento"; style="rounded,filled"; fillcolor="#E8F5EF"; color="#7B9";
    grafo [label="Grafo de conocimiento\\n{r['nodos']} nodos · {r['relaciones']} relaciones", fillcolor="#009E73", fontcolor="white"];
    idx [label="Índice vectorial\\n{len(S.repo.frags)} fragmentos ·\\n{len(S.kb.docs)} documentos", fillcolor="#009E73", fontcolor="white"];
  }}
  subgraph cluster_m {{ label="4. Módulos inteligentes"; style="rounded,filled"; fillcolor="#EAF3FB"; color="#79B";
    bus [label="Buscador y ranking\\nde especialistas\\n(puntaje por evidencia)", fillcolor="#0072B2", fontcolor="white"];
    rec [label="Recomendador\\ndocumentos relacionados y\\ncapacitación", fillcolor="#0072B2", fontcolor="white"];
    rag [label="Asistente conversacional RAG\\nrecuperar → redactar con citas\\n→ abstenerse sin evidencia", fillcolor="#0072B2", fontcolor="white"];
    viz [label="Visualización del grafo\\n(pyvis, interactivo)", fillcolor="#0072B2", fontcolor="white"];
  }}
  ui [label="5. Interfaz Streamlit\\nResumen · Especialistas · Documentos ·\\nRecomendaciones · Asistente · Grafo · Evaluación", fillcolor="#D55E00", fontcolor="white"];
  usr [label="Usuarios\\nRR. HH. · Líderes de proyecto ·\\nSoporte · Colaboradores · Dirección", shape=ellipse, style=filled, fillcolor="#F7F7F7"];
  dep [label="Despliegue\\nGitHub → Streamlit Community Cloud", shape=note, style=filled, fillcolor="#FFFFFF"];

  csv -> nlp; docs -> nlp; docs -> clf; onto -> nlp; onto -> kgb;
  nlp -> kgb; nlp -> emb; clf -> kgb [label="categoría"]; csv -> kgb;
  kgb -> grafo; emb -> idx; docs -> emb;
  grafo -> bus; grafo -> rec; grafo -> rag; grafo -> viz; idx -> rec; idx -> rag; idx -> bus [style=dashed, label="interpretación\\nsemántica"];
  bus -> ui; rec -> ui; rag -> ui; viz -> ui; ui -> usr; usr -> ui [label="consultas"];
  ui -> clf [style=dashed, color="#D55E00", fontcolor="#D55E00", label="documento nuevo\\n+ confirmación / corrección"];
  ui -> dep [style=dotted, arrowhead=none];
}}'''
    _dot("arquitectura", cod)


def ontologia(S: Sistema):
    clases = ["Empleado", "Habilidad", "Tecnología", "Certificación", "Capacitación", "Proyecto", "Incidente", "Cliente",
              "Departamento", "Documento", "Categoría", "Área"]
    etiqueta = {"Categoría": "Categoría\\nde documento", "Área": "Área de\\nconocimiento"}
    nuevas = {"Documento", "Categoría", "Área"}
    rels = [("Empleado", "Habilidad", "tiene_habilidad"), ("Empleado", "Certificación", "posee"), ("Certificación", "Habilidad", "avala"),
            ("Empleado", "Capacitación", "completó"), ("Capacitación", "Habilidad", "desarrolla"), ("Empleado", "Departamento", "pertenece_a"),
            ("Empleado", "Proyecto", "participó_en"), ("Empleado", "Incidente", "resolvió"), ("Incidente", "Proyecto", "ocurrió_en"),
            ("Incidente", "Tecnología", "involucra"), ("Proyecto", "Tecnología", "usa"), ("Proyecto", "Cliente", "solicitado_por"),
            ("Tecnología", "Habilidad", "es_un_tipo_de"), ("Habilidad", "Área", "pertenece_al_área"),
            ("Documento", "Categoría", "clasificado_como"), ("Documento", "Habilidad", "menciona"), ("Documento", "Tecnología", "menciona"),
            ("Documento", "Empleado", "respalda"), ("Documento", "Proyecto", "documenta"), ("Documento", "Incidente", "documenta"),
            ("Documento", "Certificación", "acredita")]
    nodos = "\n".join(
        f'  "{c}" [label="{etiqueta.get(c, c)}", fillcolor="{COLORES[c]}", fontcolor="{_txt(c)}"' + (', style="filled,dashed,bold", penwidth=2.4' if c in nuevas else "") + "];"
        for c in clases)
    aristas = "\n".join(f'  "{a}" -> "{b}" [label="{l}", color="{"#B24" if a == "Documento" or b in ("Categoría", "Área") else "#667"}", fontcolor="{"#B24" if a == "Documento" or b in ("Categoría", "Área") else "#445"}"];' for a, b, l in rels)
    cod = f'''digraph O {{
  layout=dot; rankdir=TB; nodesep=0.45; ranksep=0.85; fontname="{FUENTE}"; pad=0.25; bgcolor="white";
  node [shape=box, style="rounded,filled", fontname="{FUENTE}", fontsize=12, margin="0.18,0.1", color="#333", height=0.55];
  edge [fontname="{FUENTE}", fontsize=9];
{nodos}
{aristas}
  leyenda [shape=plaintext, style="", fontsize=10, label=<<table border="0" cellborder="0"><tr><td align="left">Líneas azul-gris: ontología de la primera etapa</td></tr><tr><td align="left"><font color="#B24">Rojo y nodos con borde punteado: ampliación de esta etapa</font></td></tr></table>>];
}}'''
    _dot("ontologia", cod)


def flujo_ingesta(S: Sistema):
    pasos = [("1. Carga", "Archivo .txt/.pdf\\no texto pegado", "#BBBBBB"), ("2. Clasificación", "TF-IDF + Reg. Logística\\ncategoría y probabilidades", "#0072B2"),
             ("3. Extracción NLP", "Entidades normalizadas\\n(habilidades, técnologías,\\ncertificaciones, personas…)", "#0072B2"),
             ("4. Propuestas", "Relaciones «propuesto»\\n(tiene_habilidad, posee,\\nparticipó_en, resolvió…)", "#E69F00"),
             ("5. Validación humana", "Confirmar / corregir\\ncategoría y relaciones\\n(RR. HH. o colaborador)", "#D55E00"),
             ("6. Actualización", "Grafo + índice vectorial\\n+ reentrenamiento", "#009E73"),
             ("7. Uso", "Búsqueda · Recomendación ·\\nAsistente · Visualización", "#009E73")]
    nodos = "\n".join(f'  p{i} [label="{t}\\n{d}", fillcolor="{c}", fontcolor="{"#000" if c in ("#E69F00", "#BBBBBB") else "#fff"}"];' for i, (t, d, c) in enumerate(pasos))
    cad = " -> ".join(f"p{i}" for i in range(len(pasos)))
    cod = f'''digraph F {{
  rankdir=LR; nodesep=0.3; ranksep=0.45; fontname="{FUENTE}"; pad=0.25; bgcolor="white";
  node [shape=box, style="rounded,filled", fontname="{FUENTE}", fontsize=11, margin="0.15,0.1", color="#333"];
  edge [color="#556", penwidth=1.4];
{nodos}
  {cad};
  p4 -> p1 [style=dashed, color="#D55E00", constraint=false, label="corrección →\\nreentrena", fontname="{FUENTE}", fontsize=9, fontcolor="#D55E00"];
}}'''
    _dot("flujo_ingesta", cod)


def grafo_estatico(S: Sistema):
    """Vista estática del entorno de Ana Torres y Luis Pérez (equivale al ejemplo del avance)."""
    centros = ["EMP-001", "EMP-002"]
    sub = S.kb.subgrafo(centros, 1, tipos={"Habilidad", "Certificación", "Proyecto", "Incidente", "Capacitación", "Departamento"}, max_nodos=60)
    pos = nx.spring_layout(sub.to_undirected(), seed=7, k=0.9, iterations=200)
    fig, ax = plt.subplots(figsize=(11, 7.4))
    for a, b, d in sub.edges(data=True):
        ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(arrowstyle="-|>", color="#9aa0a6", lw=0.9, shrinkA=9, shrinkB=9))
    for n, d in sub.nodes(data=True):
        t = d["tipo"]
        ax.scatter(*pos[n], s=900 if n in centros else 420, color=COLORES[t], edgecolor="#222" if n in centros else "white", linewidth=1.6, zorder=3)
        ax.text(pos[n][0], pos[n][1] - 0.075, d["label"][:26], ha="center", va="top", fontsize=7.2 if n not in centros else 9,
                fontweight="bold" if n in centros else "normal", zorder=4)
    for t in {d["tipo"] for _, d in sub.nodes(data=True)}:
        ax.scatter([], [], color=COLORES[t], label=t, s=90)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    ax.set_axis_off(); ax.set_title("Entorno de Ana Torres y Luis Pérez en el grafo de conocimiento (vista estática)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "grafo_ana_luis.png", dpi=170); plt.close(fig)


def todo():
    S = Sistema()
    arquitectura(S); ontologia(S); flujo_ingesta(S); grafo_estatico(S)
    return [p.name for p in FIG.glob("*.png")]


if __name__ == "__main__":
    print(todo())
