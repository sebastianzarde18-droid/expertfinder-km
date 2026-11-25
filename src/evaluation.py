"""Evaluación del prototipo: métricas y casos de prueba con resultados reales.

Genera en reports/: metrics.json, clasificacion_*.csv, recuperacion.csv, expertos.csv, chatbot.csv,
recomendacion_*.csv, tabla_pruebas.csv y figuras en figures/.
Ejecutar:  python -m src.evaluation
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .classifier import ClasificadorDocs
from .ontology import HABILIDADES
from .sistema import Sistema

RAIZ = Path(__file__).resolve().parent.parent
REPORTES, FIGURAS = RAIZ / "reports", RAIZ / "figures"
PALETA = {"azul": "#0072B2", "naranja": "#E69F00", "verde": "#009E73", "rojo": "#D55E00", "gris": "#7f7f7f", "rosa": "#CC79A7"}

# ----------------------------------------------------------------------------- conjuntos de prueba
# (consulta, habilidad relevante, tipo): «literal» usa el nombre exacto; «parafraseada» usa sinónimos o descripciones
CONSULTAS_RECUPERACION = [
    ("bases de datos", "Bases de datos", "literal"), ("Python", "Python", "literal"), ("machine learning", "Machine Learning", "literal"),
    ("ciberseguridad", "Ciberseguridad", "literal"), ("gestión de proyectos", "Gestión de proyectos", "literal"),
    ("pruebas de software", "Pruebas de software", "literal"), ("Java", "Java", "literal"), ("redes", "Redes", "literal"),
    ("quién sabe de SQL y administración de datos", "Bases de datos", "parafraseada"),
    ("experiencia con contenedores y despliegue continuo", "DevOps", "parafraseada"),
    ("especialista en la nube", "Computación en la nube", "parafraseada"),
    ("aprendizaje automático e inteligencia artificial", "Machine Learning", "parafraseada"),
    ("diseño de interfaces y experiencia de usuario", "Diseño UX/UI", "parafraseada"),
    ("desarrollo web front end", "JavaScript", "parafraseada"),
    ("atención a usuarios y mesa de ayuda", "Soporte técnico", "parafraseada"),
    ("tableros de inteligencia de negocios y analítica", "Análisis de datos", "parafraseada"),
]

# (consulta, función de verdad sobre ground truth) — verdad = quién realmente tiene la habilidad/tecnología según el generador
CONSULTAS_EXPERTOS = [
    ("Necesitamos una persona con experiencia en Python y bases de datos", ["Python", "Bases de datos"], None),
    ("alguien que domine py y BD", ["Python", "Bases de datos"], None),
    ("especialista en Postgres", [], "PostgreSQL"),
    ("experto en la nube", ["Computación en la nube"], None),
    ("DevOps y nube", ["DevOps", "Computación en la nube"], None),
    ("persona de ciberseguridad", ["Ciberseguridad"], None),
    ("análisis de datos con Python", ["Análisis de datos", "Python"], None),
    ("machine learning", ["Machine Learning"], None),
    ("gestión de proyectos y scrum", ["Gestión de proyectos", "Metodologías ágiles"], None),
    ("experiencia en k8s", [], "Kubernetes"),
    ("desarrollo en JavaScript", ["JavaScript"], None),
    ("alguien de pruebas QA", ["Pruebas de software"], None),
]

DEV_PREGUNTAS = [  # se usan SOLO para calibrar umbrales
    ("¿Cómo se atiende un incidente de severidad media?", True), ("¿Qué niveles de habilidad existen?", True),
    ("¿Cuál fue la solución del incidente por tablespace lleno en Oracle?", True), ("¿Qué incluye el informe de cierre de proyecto?", True),
    ("¿Quién ganó el mundial de futbol?", False), ("¿Cómo se prepara una pizza?", False),
    ("¿Qué clima hará mañana en Monterrey?", False), ("¿Cuál es la política de gastos de viaje?", False),
]

# (pregunta, ¿tiene respuesta?, fragmentos esperados en la respuesta, título de documento fuente esperado)
PREGUNTAS_TEST = [
    ("¿Cuál es el procedimiento para solicitar un especialista?", True, ["Paso 1", "2 días hábiles"], "Procedimiento para solicitar un especialista"),
    ("¿Cuántos días hábiles tiene el colaborador para confirmar las habilidades propuestas?", True, ["5 días hábiles"], "Guía de validación de habilidades propuestas"),
    ("¿Qué pasa con una certificación vencida en el ranking?", True, ["no cuenta"], "Política de vigencia y renovación de certificaciones"),
    ("¿Con cuántos días de anticipación avisa el sistema de un vencimiento de certificación?", True, ["90 días"], "Política de vigencia y renovación de certificaciones"),
    ("¿Quién resolvió el incidente INC-0457 y cómo?", True, ["Luis Pérez", "índices"], "Ticket INC-0457"),
    ("¿Cuál fue la causa del incidente INC-0457?", True, ["saturación de conexiones"], "Ticket INC-0457"),
    ("¿Qué certificaciones tiene Ana Torres?", True, ["PCAP"], "CV de Ana Torres"),
    ("¿En cuánto tiempo se elabora el informe de cierre de un proyecto?", True, ["10 días hábiles"], "Procedimiento de cierre de proyecto y lecciones aprendidas"),
    ("¿Quién tiene experiencia en Kubernetes?", True, ["Javier Salinas"], None),
    ("¿Cuántas horas de capacitación al año financia la empresa?", True, ["40 horas"], "Política de capacitación"),
    ("¿En cuánto tiempo se escala un incidente de severidad alta?", True, ["30 minutos"], "Procedimiento de atención y escalamiento de incidentes"),
    ("¿Qué incidentes ha habido con Kubernetes y cómo se resolvieron?", True, ["INC-0505", "manifiesto"], None),
    ("¿Qué norma regula el tratamiento de los datos personales de los colaboradores?", True, ["LFPDPPP"], "Política de privacidad y protección de datos personales"),
    ("¿Cuál es el precio del dólar hoy?", False, [], None),
    ("¿Cuál es el procedimiento para solicitar un equipo de cómputo nuevo?", False, [], None),
    ("¿Quién sabe de Rust?", False, [], None),
    ("¿Cuál es el salario de Ana Torres?", False, [], None),
    ("¿Qué proyectos tiene el cliente Banco Sur?", False, [], None),
    ("¿Cuál es la política de trabajo remoto?", False, [], None),
]

CASOS_RECOMENDACION = [
    ("Perfil de Daniel: Python básico y SQL; objetivo: especializarse en análisis de datos", {"Python": 2, "Bases de datos": 2}, "quiere especializarse en análisis de datos", set(),
     {"K06", "K07", "K08"}),
    ("Perfil de redes (nivel 3) que quiere aprender nube y DevOps", {"Redes": 3}, "quiero aprender computación en la nube y DevOps", set(), {"K12", "K13", "K10"}),
    ("Ana Torres (Python 4, BD 3) que quiere aprender Machine Learning", {"Python": 4, "Bases de datos": 3}, "interesada en Machine Learning", {"K05"}, {"K07", "K24"}),
]


def _fig(nombre):
    FIGURAS.mkdir(exist_ok=True)
    return FIGURAS / nombre


def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


# ----------------------------------------------------------------------------- 1. clasificación
def eval_clasificacion(S: Sistema) -> dict:
    X = [d["texto"] for d in S.kb.docs if d["estado"] == "confirmado"]
    y = [d["categoria"] for d in S.kb.docs if d["estado"] == "confirmado"]
    ev = ClasificadorDocs.evaluar(X, y, k=5, semilla=0)
    ev["comparacion"].round(4).to_csv(REPORTES / "clasificacion_modelos.csv", index=False)
    rep = ev["reporte"].round(4)
    rep.to_csv(REPORTES / "clasificacion_reporte.csv")
    cm = pd.DataFrame(ev["matriz"], index=ev["clases"], columns=ev["clases"])
    cm.to_csv(REPORTES / "clasificacion_matriz.csv")
    # figura: matriz de confusión
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(ev["matriz"], cmap="Blues")
    corto = [c.replace(" de ", " ").replace(" o ", "/")[:22] for c in ev["clases"]]
    ax.set_xticks(range(len(corto)), corto, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(corto)), corto, fontsize=8)
    for i in range(len(corto)):
        for j in range(len(corto)):
            v = ev["matriz"][i, j]
            ax.text(j, i, v, ha="center", va="center", color="white" if v > ev["matriz"].max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Categoría predicha"); ax.set_ylabel("Categoría real")
    ax.set_title("Matriz de confusión (validación cruzada de 5 particiones, 168 documentos)", fontsize=10)
    fig.tight_layout(); fig.savefig(_fig("matriz_confusion.png"), dpi=180); plt.close(fig)
    # partición fija 75/25 como segunda verificación
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    clf = ClasificadorDocs().entrenar(Xtr, ytr)
    pred = [clf.predecir(t)[0] for t in Xte]
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    errores = [dict(titulo=S.kb.docs[i]["titulo"], real=ev["y_true"][i], predicha=ev["y_pred"][i]) for i in [e["indice"] for e in ev["errores"]]]
    pd.DataFrame(errores).to_csv(REPORTES / "clasificacion_errores.csv", index=False)
    m = ev["comparacion"].set_index("modelo").loc["logreg"]
    return dict(n_documentos=len(X), cv_accuracy=round(float(m.accuracy), 4), cv_precision_macro=round(float(m.precision_macro), 4),
                cv_recall_macro=round(float(m.recall_macro), 4), cv_f1_macro=round(float(m.f1_macro), 4),
                holdout_test=len(yte), holdout_precision_macro=round(float(p), 4), holdout_recall_macro=round(float(r), 4),
                holdout_f1_macro=round(float(f), 4), errores=errores,
                comparacion=ev["comparacion"].round(4).to_dict("records"))


# ----------------------------------------------------------------------------- 2. recuperación de documentos
def eval_recuperacion(S: Sistema) -> dict:
    relev = {h: {d["id"] for d in S.kb.docs if h in d.get("temas", [])} for h in HABILIDADES}
    modos = {"Palabras clave (exacta)": ("palabras", False), "Semántica LSA": ("semantica", False),
             "Semántica + ontología": ("semantica", True), "Híbrida + ontología": ("hibrida", True)}
    filas = []
    for q, h, tipo in CONSULTAS_RECUPERACION:
        R = relev[h]
        for nombre, (modo, exp) in modos.items():
            res = [r["doc_id"] for r in S.repo.buscar_documentos(q, k=10, modo=modo, expandir=exp)]
            hit = lambda k: len(set(res[:k]) & R)
            filas.append(dict(consulta=q, tipo=tipo, modo=nombre, relevantes=len(R), devueltos=len(res),
                              P_at_5=hit(5) / 5, P_at_10=hit(10) / 10, R_at_10=hit(10) / len(R),
                              R_precision=len(set(res[:len(R)]) & R) / len(R)))
    df = pd.DataFrame(filas)
    df.round(4).to_csv(REPORTES / "recuperacion_detalle.csv", index=False)
    resumen = df.groupby("modo")[["P_at_5", "P_at_10", "R_at_10", "R_precision"]].mean().loc[list(modos)]
    por_tipo = df.groupby(["tipo", "modo"])[["P_at_5", "R_at_10"]].mean().round(3)
    resumen.round(4).to_csv(REPORTES / "recuperacion_resumen.csv")
    por_tipo.to_csv(REPORTES / "recuperacion_por_tipo.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(resumen)); w = 0.2
    for i, (col, lab, c) in enumerate([("P_at_5", "Precision@5", PALETA["azul"]), ("P_at_10", "Precision@10", PALETA["verde"]),
                                       ("R_at_10", "Recall@10", PALETA["naranja"]), ("R_precision", "R-Precision", PALETA["rojo"])]):
        b = ax.bar(x + (i - 1.5) * w, resumen[col], w, label=lab, color=c)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [n.replace(" + ", "\n+ ") for n in resumen.index], fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Valor promedio (16 consultas)"); _estilo(ax)
    ax.set_title("Búsqueda por palabras clave vs. semántica", fontsize=10); ax.legend(fontsize=8, ncol=4, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(_fig("comparacion_busqueda.png"), dpi=180); plt.close(fig)
    return dict(resumen=resumen.round(4).to_dict("index"), por_tipo={f"{a}|{b}": v for (a, b), v in por_tipo.to_dict("index").items()}, n_consultas=len(CONSULTAS_RECUPERACION))


# ----------------------------------------------------------------------------- 3. búsqueda de expertos
def eval_expertos(S: Sistema) -> dict:
    gt = json.load(open(S.kb.data_dir / "ground_truth.json", encoding="utf-8"))
    reales, tecn = gt["habilidades_reales"], gt["tecnologias_empleado"]
    filas = []
    for q, habs, tech in CONSULTAS_EXPERTOS:
        if tech:
            verdad = {e for e, ts in tecn.items() if tech in ts}
        else:
            verdad = {e for e, hs in reales.items() if all(h in hs for h in habs)}
        t = time.perf_counter()
        r = S.buscador.buscar(q, k=10)
        ms = (time.perf_counter() - t) * 1000
        ids = [x["empleado"] for x in r["resultados"]]
        completos = [x["empleado"] for x in r["resultados"] if x["cubiertas"] == len(r["requisito"].habilidades)]
        k3 = ids[:3]
        filas.append(dict(consulta=q, interpretado=", ".join(r["requisito"].habilidades), verdad=len(verdad), devueltos=len(ids),
                          P_at_3=(len(set(k3) & verdad) / len(k3)) if k3 else 0.0,
                          cobertura_R10=len(set(ids) & verdad) / len(verdad) if verdad else np.nan,
                          pct_top3_cumple_todos=(len(set(k3) & set(completos)) / len(k3)) if k3 else 0.0, ms=round(ms, 2)))
    df = pd.DataFrame(filas)
    df.round(3).to_csv(REPORTES / "expertos_detalle.csv", index=False)
    return dict(P_at_3=round(float(df.P_at_3.mean()), 4), cobertura_R10=round(float(df.cobertura_R10.mean()), 4),
                tiempo_ms_promedio=round(float(df.ms.mean()), 2), tiempo_ms_max=round(float(df.ms.max()), 2), n_consultas=len(df),
                consultas_sin_resultado=int((df.devueltos == 0).sum()))


# ----------------------------------------------------------------------------- 4. recomendación
def eval_recomendacion(S: Sistema) -> dict:
    # (a) documentos relacionados
    muestra = [d for d in S.kb.docs if d["categoria"] in ("Ticket de incidente", "Informe de cierre de proyecto", "Certificado o constancia", "Minuta de reunión")]
    muestra = muestra[::6][:20]
    filas = []
    for d in muestra:
        rel = S.recomendador.relacionados(d["id"], k=3)
        ok = [bool(set(S.kb.documento(r["doc_id"]).get("temas", [])) & set(d["temas"])) for r in rel]
        filas.append(dict(doc=d["titulo"], P_at_3=sum(ok) / len(ok), similitud_media=np.mean([r["similitud_coseno"] for r in rel])))
    df = pd.DataFrame(filas)
    df.round(3).to_csv(REPORTES / "recomendacion_documentos.csv", index=False)
    # (b) capacitación
    filas2 = []
    for desc, hab, obj, comp, esperado in CASOS_RECOMENDACION:
        objs = S.recomendador.objetivo_desde_texto(obj)
        rec = S.recomendador.capacitacion(hab, objs, comp, k=5)
        ids = [r["curso_id"] for r in rec]
        dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(ids) if c in esperado)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(esperado), 5)))
        filas2.append(dict(caso=desc, objetivo_detectado=", ".join(objs), recomendados=" | ".join(r["curso"] for r in rec),
                           esperados=", ".join(sorted(esperado)), hit_at_5=len(set(ids) & esperado) / len(esperado),
                           P_at_5=len(set(ids) & esperado) / max(1, len(ids)), NDCG_at_5=dcg / idcg if idcg else 0,
                           sin_repetidos=not (set(ids) & comp)))
    df2 = pd.DataFrame(filas2)
    df2.round(3).to_csv(REPORTES / "recomendacion_capacitacion.csv", index=False)
    return dict(docs_P_at_3=round(float(df.P_at_3.mean()), 4), docs_similitud_media=round(float(df.similitud_media.mean()), 4), docs_n=len(df),
                cap_hit_at_5=round(float(df2.hit_at_5.mean()), 4), cap_NDCG_at_5=round(float(df2.NDCG_at_5.mean()), 4), cap_casos=len(df2),
                cap_detalle=df2.round(3).to_dict("records"))


# ----------------------------------------------------------------------------- 5. chatbot
def eval_chatbot(S: Sistema) -> dict:
    from . import chatbot as cb
    # calibración con el conjunto de desarrollo (sin usar las preguntas de prueba)
    barrido = []
    for tp in (0.10, 0.15, 0.20, 0.25, 0.30):
        for ts in (0.4, 0.5, 0.6, 0.65, 0.7, 0.8):
            cb.UMBRAL_PUNTAJE, cb.UMBRAL_COBERTURA = tp, ts
            ok = sum(S.asistente.responder(q)["suficiente"] == esperado for q, esperado in DEV_PREGUNTAS)
            barrido.append(dict(umbral_puntaje=tp, umbral_soporte=ts, aciertos=ok, de=len(DEV_PREGUNTAS)))
    pd.DataFrame(barrido).to_csv(REPORTES / "chatbot_calibracion_dev.csv", index=False)
    cb.UMBRAL_PUNTAJE, cb.UMBRAL_COBERTURA = 0.20, 0.65  # valores fijados antes de medir en el conjunto de prueba
    filas, tiempos = [], []
    for q, responde, esperados, fuente in PREGUNTAS_TEST:
        t = time.perf_counter()
        r = S.asistente.responder(q)
        tiempos.append(time.perf_counter() - t)
        txt = r["respuesta"]
        if responde:
            correcta = r["suficiente"] and all(e.lower() in txt.lower() for e in esperados)
            fuente_ok = None
            if fuente:
                ids = {d["id"] for d in S.kb.docs if d["titulo"] == fuente}
                fuente_ok = bool(ids & set(r["fuentes"]))
            else:
                fuente_ok = bool(r["fuentes"])
            resultado = "correcta" if correcta else ("falso rechazo" if not r["suficiente"] else "incorrecta")
        else:
            correcta, fuente_ok = (not r["suficiente"]), None
            resultado = "rechazo correcto" if correcta else "respuesta inventada"
        filas.append(dict(pregunta=q, tiene_respuesta=responde, resultado=resultado, fuentes=" ".join(r["fuentes"]),
                          fundamentada=fuente_ok, respuesta=txt.replace("\n", " ")[:400], ms=round(tiempos[-1] * 1000, 1)))
    df = pd.DataFrame(filas)
    df.to_csv(REPORTES / "chatbot_detalle.csv", index=False)
    con = df[df.tiene_respuesta]
    sin = df[~df.tiene_respuesta]
    res = dict(preguntas=len(df), con_respuesta=len(con), sin_respuesta=len(sin),
               exactitud_con_respuesta=round(float((con.resultado == "correcta").mean()), 4),
               fundamentacion=round(float(con.fundamentada.astype(bool).mean()), 4),
               rechazo_correcto_sin_respuesta=round(float((sin.resultado == "rechazo correcto").mean()), 4),
               porcentaje_respuestas_correctas=round(float(df.resultado.isin(["correcta", "rechazo correcto"]).mean()), 4),
               falsos_rechazos=int((df.resultado == "falso rechazo").sum()), respuestas_inventadas=int((df.resultado == "respuesta inventada").sum()),
               tiempo_ms_promedio=round(float(df.ms.mean()), 1), modo=S.asistente.modo,
               dev_mejor_config=max(barrido, key=lambda b: b["aciertos"]))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cuentas = df.resultado.value_counts()
    colores = {"correcta": PALETA["verde"], "rechazo correcto": PALETA["azul"], "falso rechazo": PALETA["naranja"],
               "incorrecta": PALETA["rojo"], "respuesta inventada": PALETA["rojo"]}
    b = ax.barh(cuentas.index, cuentas.values, color=[colores[c] for c in cuentas.index])
    for r in b:
        ax.text(r.get_width() + 0.1, r.get_y() + r.get_height() / 2, int(r.get_width()), va="center", fontsize=9)
    ax.set_xlabel("Número de preguntas"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Resultado del asistente en {len(df)} preguntas de prueba", fontsize=10)
    fig.tight_layout(); fig.savefig(_fig("chatbot_resultados.png"), dpi=180); plt.close(fig)
    return res


# ----------------------------------------------------------------------------- 6. tabla de pruebas integradas
def tabla_pruebas(S: Sistema) -> pd.DataFrame:
    filas = []

    def reg(n, tipo, entrada, esperado, obtenido, ok):
        filas.append({"Prueba": n, "Tipo": tipo, "Entrada o consulta": entrada, "Resultado esperado": esperado,
                      "Resultado obtenido": obtenido, "¿Cumple?": "Sí" if ok else "No"})

    q1 = "Necesitamos una persona con experiencia en Python y bases de datos"
    r1 = S.buscador.buscar(q1, k=5)
    top = r1["resultados"]
    reg(1, "Consulta válida", q1, "Especialistas con 2 de 2 requisitos y evidencias",
        "; ".join(f"{x['nombre']} ({x['cobertura']}, {x['puntaje']})" for x in top[:3]) + f" · {r1['tiempo_ms']} ms",
        bool(top) and top[0]["cobertura"] == "2 de 2" and all(d["evidencias"] for d in top[0]["detalle"].values()))
    q2 = "alguien que domine py y BD"
    r2 = S.buscador.buscar(q2, k=5)
    a, b = {x["empleado"] for x in r1["resultados"]}, {x["empleado"] for x in r2["resultados"]}
    reg(2, "Consulta similar (sinónimos)", q2, "Mismos especialistas que la prueba 1 (sinónimos normalizados)",
        f"Interpretado como: {', '.join(r2['requisito'].habilidades)}; coincidencia con prueba 1: {len(a & b)}/{len(a)} en el top-5", len(a & b) >= 4)
    q3 = "necesito a alguien de datos"
    r3 = S.buscador.buscar(q3, k=5)
    reg(3, "Consulta ambigua", q3, "Resultado justificable: indica cómo interpretó la consulta o pide precisarla",
        f"Interpretación: «{r3['requisito'].interpretacion}» → {', '.join(r3['requisito'].habilidades) or 'ninguna'}; " +
        (f"primeros: {', '.join(x['nombre'] for x in r3['resultados'][:3])}" if r3["resultados"] else "sin resultados"),
        bool(r3["requisito"].interpretacion) and (bool(r3["resultados"]) or not r3["requisito"].habilidades))
    q4 = "especialista en Rust"
    r4 = S.buscador.buscar(q4)
    reg(4, "Información inexistente", q4, "Mensaje de que no hay datos; sin resultados inventados",
        f"{len(r4['resultados'])} resultados; «{r4['requisito'].interpretacion}»", len(r4["resultados"]) == 0)
    q5 = "Kubernetes con certificación vigente y experiencia en Docker, AWS y bases de datos"
    r5 = S.buscador.buscar(q5, k=5)
    ivan = next((x for x in S.buscador.buscar("Kubernetes", k=30)["resultados"] if x["nombre"] == "Iván Cavazos"), None)
    venc = ivan and any(d["vencidas"] for d in ivan["detalle"].values())
    reg(5, "Caso de mayor complejidad", q5, "Ranking por cobertura de varios requisitos; certificación vencida no suma (R2)",
        f"Requisitos: {', '.join(r5['requisito'].habilidades)}; top: " + "; ".join(f"{x['nombre']} ({x['cobertura']}, {x['puntaje']})" for x in r5["resultados"][:3]) +
        (f". Iván Cavazos: certificación CKA vencida detectada y excluida del puntaje" if venc else ""),
        bool(r5["resultados"]) and bool(venc))
    # chatbot
    r6 = S.asistente.responder("¿Quién resolvió el incidente INC-0457 y cómo?")
    reg(6, "Asistente: pregunta con respuesta", "¿Quién resolvió el incidente INC-0457 y cómo?", "Luis Pérez, con cita del ticket",
        r6["respuesta"][:230], "Luis Pérez" in r6["respuesta"] and bool(r6["fuentes"]))
    r7 = S.asistente.responder("¿Cuál es el procedimiento para solicitar un equipo de cómputo nuevo?")
    reg(7, "Asistente: sin información documental", "¿Cuál es el procedimiento para solicitar un equipo de cómputo nuevo?",
        "Reconoce que no hay información y no inventa", r7["respuesta"][:200], not r7["suficiente"])
    # clasificador
    txt = ("TICKET INC-0999. Severidad: alta. Proyecto afectado: Portal Bancario. Síntoma: la aplicación no responde. "
           "Causa raíz: saturación de conexiones en PostgreSQL. Solución aplicada: reinicio del pool y ajuste de índices. Responsable: Luis Pérez. Tiempo de resolución: 2 horas.")
    cat, pr = S.clasificador.predecir(txt)
    reg(8, "Clasificación de documento nuevo", "Texto de un ticket de incidente (INC-0999)", "Categoría «Ticket de incidente»",
        f"{cat} ({max(pr.values()):.0%})", cat == "Ticket de incidente")
    # recomendación
    rel = S.recomendador.relacionados(S.kb.doc_de(incidente="INC-0457"), k=3)
    reg(9, "Recomendación de documentos", "Documentos relacionados con el ticket INC-0457", "Documentos sobre PostgreSQL / bases de datos con similitud mostrada",
        "; ".join(f"{r['titulo']} (cos {r['similitud_coseno']})" for r in rel),
        all("Bases de datos" in S.kb.documento(r["doc_id"]).get("temas", []) for r in rel))
    # casos en los que el sistema puede fallar
    q10 = "¿Qué norma regula el tratamiento de los datos personales de los colaboradores?"
    r10 = S.asistente.responder(q10)
    reg(10, "Asistente: pregunta parafraseada", q10, "«LFPDPPP» citando la política de privacidad",
        r10["respuesta"][:200], "LFPDPPP" in r10["respuesta"])
    from .classifier import UMBRAL_CONFIANZA
    t11 = "Informe mensual de avances: el proyecto Portal Bancario avanzó 80%, con retraso en pruebas."
    cat11, pr11 = S.clasificador.predecir(t11)
    conf = max(pr11.values())
    reg(11, "Clasificación: documento de tipo no entrenado", "Informe mensual de avances (categoría ausente del entrenamiento)",
        "Marca baja confianza para revisión manual (no fuerza una categoría con falsa seguridad)",
        f"{cat11} ({conf:.0%}); " + ("se solicita revisión manual" if conf < UMBRAL_CONFIANZA else "sin alerta"), conf < UMBRAL_CONFIANZA)
    return pd.DataFrame(filas)


def ejecutar() -> dict:
    REPORTES.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    S = Sistema()
    m = dict(sistema=dict(nodos=S.kb.resumen()["nodos"], relaciones=S.kb.resumen()["relaciones"], documentos=len(S.kb.docs),
                          empleados=len(S.kb.empleados), tiempo_carga_s=S.tiempo_carga_s, embeddings=S.repo.emb.nombre, fragmentos=len(S.repo.frags)))
    m["clasificacion"] = eval_clasificacion(S)
    m["recuperacion"] = eval_recuperacion(S)
    m["expertos"] = eval_expertos(S)
    m["recomendacion"] = eval_recomendacion(S)
    m["chatbot"] = eval_chatbot(S)
    tp = tabla_pruebas(S)
    tp.to_csv(REPORTES / "tabla_pruebas.csv", index=False)
    m["pruebas_integradas"] = dict(total=len(tp), cumplen=int((tp["¿Cumple?"] == "Sí").sum()))
    m["tiempo_evaluacion_s"] = round(time.perf_counter() - t0, 1)
    with open(REPORTES / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1, default=str)
    return m


if __name__ == "__main__":
    m = ejecutar()
    print(json.dumps({k: v for k, v in m.items() if k not in ("clasificacion",)}, ensure_ascii=False, indent=1, default=str)[:6000])
    print(json.dumps({k: v for k, v in m["clasificacion"].items() if k not in ("errores", "comparacion")}, ensure_ascii=False))
