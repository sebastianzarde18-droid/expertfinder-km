"""ExpertFinder · Directorio inteligente de conocimiento y talento (TalentTech Consulting).

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations
import json
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.classifier import UMBRAL_CONFIANZA
from src.knowledge_graph import COLORES
from src.ontology import CATEGORIAS_DOC, CERTIFICACIONES, CURSOS, FECHA_REFERENCIA, HABILIDADES, REGLAS
from src.sistema import Sistema

RAIZ = Path(__file__).resolve().parent
st.set_page_config(page_title="ExpertFinder · TalentTech", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.ef-chip {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.78rem; margin-right:6px; color:#fff;}
.ef-ok {background:#009E73;} .ef-warn {background:#B26A00;} .ef-bad {background:#B53B00;} .ef-info {background:#0072B2;}
.ef-small {font-size:0.85rem; color:#555;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ estado
def sistema() -> Sistema:
    if "S" not in st.session_state:
        with st.spinner("Cargando base de conocimiento, grafo e índices…"):
            st.session_state.S = Sistema()
    return st.session_state.S


def chip(texto: str, clase: str) -> str:
    return f'<span class="ef-chip {clase}">{texto}</span>'


def etiqueta(S: Sistema, i: str) -> str:
    if i in S.kb.G:
        return S.kb.G.nodes[i]["label"]
    return i


def estado_chip(estado: str) -> str:
    return chip(estado, {"comprobada": "ef-ok", "declarada": "ef-warn", "inferida": "ef-info"}.get(estado, "ef-bad"))


def mostrar_grafo(S: Sistema, G, altura: int = 560, resaltar=None):
    if G.number_of_nodes() == 0:
        st.info("No hay nodos que mostrar con los filtros elegidos.")
        return
    html = S.kb.a_html(G, f"{altura - 20}px", resaltar)
    if hasattr(st, "iframe"):
        st.iframe(html, height=altura)
    else:  # versiones anteriores de Streamlit
        components.html(html, height=altura, scrolling=False)


def guardar_retro(consulta: str, empleado: str, util: bool):
    try:
        f = RAIZ / "data" / "feedback.csv"
        nuevo = not f.exists()
        with open(f, "a", encoding="utf-8") as fh:
            if nuevo:
                fh.write("fecha,consulta,empleado,util\n")
            fh.write(f'{datetime.now().isoformat(timespec="seconds")},"{consulta}",{empleado},{int(util)}\n')
    except OSError:
        pass
    st.session_state.setdefault("retro", []).append((consulta, empleado, util))


S = sistema()

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🧭 ExpertFinder")
    st.caption("Directorio inteligente de conocimiento y talento · TalentTech Consulting (datos simulados)")
    fecha = st.date_input("Fecha de referencia (vigencia de certificaciones)", value=S.kb.fecha_ref, key="fecha_ref")
    if fecha != S.kb.fecha_ref:
        S.cambiar_fecha(fecha)
        st.rerun()
    with st.expander("Pesos del ranking (ajustables)"):
        p = S.buscador.pesos
        p["certificacion"] = st.slider("Certificación vigente", 0.0, 0.6, p["certificacion"], 0.05)
        p["proyecto"] = st.slider("Por proyecto con la tecnología", 0.0, 0.3, p["proyecto"], 0.02)
        p["incidente"] = st.slider("Por incidente resuelto", 0.0, 0.3, p["incidente"], 0.02)
        p["declarada"] = st.slider("Habilidad declarada", 0.0, 0.4, p["declarada"], 0.05)
    st.markdown("**Componentes activos**")
    st.caption(f"Embeddings: {S.repo.emb.nombre}")
    st.caption(f"Asistente: {S.asistente.modo}")
    st.caption(f"Clasificador: TF-IDF + Regresión Logística ({len(S.clasificador.base_x) + len(S.clasificador.extra_x)} documentos)")
    with st.expander("Reglas de coherencia"):
        for k, v in REGLAS.items():
            st.markdown(f"**{k}.** {v}")
    if st.button("Reiniciar datos de la sesión"):
        for k in list(st.session_state):
            del st.session_state[k]
        st.rerun()

tabs = st.tabs(["🏠 Resumen", "🔎 Especialistas", "📄 Documentos", "🎯 Recomendaciones", "💬 Asistente", "🕸️ Grafo", "📊 Evaluación"])

# ================================================================== RESUMEN
with tabs[0]:
    res = S.kb.resumen()
    st.header("Del conocimiento disperso al conocimiento consultable")
    st.write("ExpertFinder integra **clasificación automática de documentos**, **búsqueda de especialistas con evidencia**, "
             "**recomendación**, un **asistente conversacional con fuentes** y un **grafo de conocimiento** sobre un mismo repositorio.")
    c = st.columns(5)
    c[0].metric("Empleados", len(S.kb.empleados))
    c[1].metric("Documentos", len(S.kb.docs))
    c[2].metric("Nodos del grafo", res["nodos"])
    c[3].metric("Relaciones", res["relaciones"])
    c[4].metric("Habilidades", len(HABILIDADES))
    izq, der = st.columns([1.1, 1])
    with izq:
        st.subheader("Brechas de conocimiento")
        st.caption("Expertos con la habilidad *comprobada* (con evidencia vigente). Las habilidades con pocos expertos son riesgo de dependencia.")
        br = S.kb.brechas()
        st.bar_chart(br.set_index("habilidad")["expertos_comprobados"], horizontal=True, height=420)
    with der:
        st.subheader("Certificaciones por vencer o vencidas")
        ce = S.kb.certificaciones_estado(90)
        ce = ce[ce.estado != "vigente"]
        if ce.empty:
            st.success("No hay certificaciones vencidas ni por vencer en los próximos 90 días.")
        else:
            st.dataframe(ce.rename(columns={"dias": "días"}), hide_index=True, width="stretch")
        st.subheader("Habilidades con un solo experto")
        uno = br[br.expertos_comprobados <= 1]
        st.dataframe(uno[["habilidad", "expertos_comprobados", "expertos"]], hide_index=True, width="stretch")
    arq = RAIZ / "figures" / "arquitectura.png"
    if arq.exists():
        with st.expander("Arquitectura del sistema"):
            st.image(str(arq))

# ================================================================== ESPECIALISTAS
with tabs[1]:
    st.header("Buscar especialistas")
    st.caption("Escriba el requisito en lenguaje natural. Se normalizan sinónimos («BD», «Postgres», «k8s») y se muestra la evidencia de cada resultado.")
    ej = st.columns(4)
    ejemplos = ["Necesitamos una persona con experiencia en Python y bases de datos", "experto en Kubernetes", "análisis de datos con Python", "especialista en ciberseguridad y redes"]
    for i, e in enumerate(ejemplos):
        if ej[i].button(e, key=f"ej{i}", width="stretch"):
            st.session_state["q_exp"] = e
    consulta = st.text_input("Consulta", key="q_exp", placeholder="Ej.: persona con experiencia en Python y bases de datos")
    f1, f2, f3, f4 = st.columns(4)
    solo_disp = f1.checkbox("Excluir personal ocupado")
    dep = f2.selectbox("Departamento", ["(todos)"] + sorted(S.kb.empleados.departamento.unique()))
    nivel = f3.slider("Nivel mínimo declarado", 0, 5, 0)
    topk = f4.slider("Resultados", 3, 15, 8)
    if consulta.strip():
        try:
            r = S.buscador.buscar(consulta, k=topk, solo_disponibles=solo_disp, departamento=None if dep == "(todos)" else dep, nivel_min=nivel)
        except Exception as ex:  # noqa: BLE001
            st.error(f"No fue posible procesar la consulta ({type(ex).__name__}). Intente reformularla.")
            r = None
        if r:
            req = r["requisito"]
            if req.metodo == "ninguna":
                st.warning(req.interpretacion + " Pruebe con un término del catálogo (por ejemplo: Python, bases de datos, nube, DevOps).")
            else:
                (st.info if req.metodo in ("exacta", "aproximada") else st.warning)(
                    f"**Requisitos detectados:** {', '.join(req.habilidades)}" + (f" · tecnologías: {', '.join(req.tecnologias)}" if req.tecnologias else "") +
                    f"\n\n{req.interpretacion}")
                if not r["resultados"]:
                    st.warning("No hay empleados con evidencia para esos requisitos y filtros. Relaje los filtros o amplíe la búsqueda.")
                else:
                    st.caption(f"{r['total']} candidatos · {r['tiempo_ms']} ms")
                    tabla = pd.DataFrame([dict(Especialista=x["nombre"], Puesto=x["puesto"], Cobertura=x["cobertura"], Puntaje=x["puntaje"],
                                               Disponibilidad=x["disponibilidad"]) for x in r["resultados"]])
                    st.dataframe(tabla, hide_index=True, width="stretch",
                                 column_config={"Puntaje": st.column_config.ProgressColumn("Puntaje", min_value=0, max_value=100, format="%.1f")})
                    for x in r["resultados"]:
                        with st.expander(f"{x['nombre']} — {x['puesto']} · cobertura {x['cobertura']} · puntaje {x['puntaje']}"):
                            st.markdown(f"**Por qué es relevante:** {x['explicacion']}")
                            for h, d in x["detalle"].items():
                                st.markdown(f"**{h}** {estado_chip(d['estado'])} puntaje parcial {d['puntaje']:.2f}", unsafe_allow_html=True)
                                filas = [dict(Tipo=e["tipo"], Evidencia=e["texto"], Documento=e["doc"] or "—") for e in d["evidencias"]]
                                filas += [dict(Tipo="certificación vencida (no suma)", Evidencia=e["texto"], Documento=e["doc"] or "—") for e in d["vencidas"]]
                                if filas:
                                    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
                                else:
                                    st.caption("Sin evidencia registrada para esta habilidad.")
                            b1, b2, _ = st.columns([1, 1, 4])
                            if b1.button("👍 Útil", key=f"u_{x['empleado']}_{consulta}"):
                                guardar_retro(consulta, x["empleado"], True)
                                st.toast("Gracias: retroalimentación registrada.")
                            if b2.button("👎 No útil", key=f"n_{x['empleado']}_{consulta}"):
                                guardar_retro(consulta, x["empleado"], False)
                                st.toast("Gracias: retroalimentación registrada.")
                    top = r["resultados"][0]
                    with st.expander(f"Ver relaciones de {top['nombre']} en el grafo"):
                        mostrar_grafo(S, S.kb.subgrafo(top["empleado"], 1, tipos={"Habilidad", "Certificación", "Proyecto", "Incidente", "Capacitación", "Departamento", "Tecnología"}), 480, {top["empleado"]})
    else:
        st.info("Escriba una consulta o use un ejemplo para comenzar.")
    if st.session_state.get("retro"):
        util = sum(1 for *_, u in st.session_state["retro"] if u)
        st.caption(f"Retroalimentación de esta sesión: {util} útiles de {len(st.session_state['retro'])} valoraciones.")

# ================================================================== DOCUMENTOS
with tabs[2]:
    st.header("Documentos: clasificar, extraer entidades y enriquecer el grafo")
    izq, der = st.columns([1.1, 1])
    with izq:
        st.subheader("1. Cargar o pegar un documento")
        archivo = st.file_uploader("Archivo .txt o .pdf", type=["txt", "pdf"])
        ejemplos_doc = {
            "(escribir o pegar)": "",
            "Ejemplo: CV de una persona nueva": "CURRICULUM VITAE\nNombre: Valeria Montoya. Puesto actual: Analista de datos. Resumen: experiencia en Python, bases de datos y Power BI. Certificaciones: Machine Learning Specialization. Formación continua: Introducción a Machine Learning. Proyectos recientes: Modelo de Predicción de Demanda (analista de datos).",
            "Ejemplo: ticket de incidente": "TICKET DE INCIDENTE\nINC-0999. Severidad: alta. Proyecto afectado: Portal Bancario. Fecha: 10/11/2026.\nSíntoma: la aplicación no responde. Tecnología involucrada: Postgres.\nCausa raíz: saturación de conexiones.\nSolución aplicada: ajuste del pool y de índices.\nResponsable: Luis Pérez. Tiempo de resolución: 2 horas.",
            "Ejemplo: certificado": "CONSTANCIA DE CERTIFICACIÓN\nCisco hace constar que Daniela Ibarra aprobó el examen de la certificación «Cisco CCNA». Fecha de obtención: 05/11/2026. Vigencia hasta: 05/11/2029. La certificación avala conocimientos de Redes.",
        }
        ej_sel = st.selectbox("Usar un ejemplo", list(ejemplos_doc))
        texto_inicial = ejemplos_doc[ej_sel]
        if archivo is not None:
            try:
                if archivo.name.lower().endswith(".pdf"):
                    import pdfplumber
                    with pdfplumber.open(archivo) as pdf:
                        texto_inicial = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
                else:
                    texto_inicial = archivo.read().decode("utf-8", errors="ignore")
            except Exception as ex:  # noqa: BLE001
                st.error(f"No se pudo leer el archivo ({type(ex).__name__}). Pegue el texto manualmente.")
        titulo = st.text_input("Título del documento", value=(archivo.name if archivo else ej_sel.replace("Ejemplo: ", "") if ej_sel != "(escribir o pegar)" else ""))
        texto = st.text_area("Texto del documento", value=texto_inicial, height=230, key=f"txt_{ej_sel}_{archivo.name if archivo else ''}")
        if st.button("🔍 Analizar documento", type="primary"):
            if len(texto.split()) < 8:
                st.warning("El documento es muy corto para clasificarlo (mínimo ~8 palabras).")
            else:
                st.session_state["analisis"] = dict(texto=texto, titulo=titulo or "Documento sin título", **S.analizar_documento(texto))
    with der:
        st.subheader("2. Resultado del análisis")
        a = st.session_state.get("analisis")
        if not a:
            st.info("Aquí aparecerá la categoría propuesta, las entidades detectadas y las relaciones sugeridas para el grafo.")
        else:
            conf = max(a["probabilidades"].values())
            st.markdown(f"**Categoría propuesta:** {a['categoria']} ({conf:.0%})")
            if conf < UMBRAL_CONFIANZA:
                st.warning("Confianza baja: revise la categoría antes de registrar el documento.")
            st.bar_chart(pd.Series(a["probabilidades"]).sort_values(), horizontal=True, height=210)
            cat_final = st.selectbox("Corregir la categoría (opcional)", CATEGORIAS_DOC, index=CATEGORIAS_DOC.index(a["categoria"]))
            if cat_final != a["categoria"]:
                _, props = S.kb.proponer_relaciones(a["texto"], cat_final)
            else:
                props = a["propuestas"]
            ents = pd.DataFrame(a["entidades"])
            if len(ents):
                st.markdown("**Entidades detectadas (NLP, normalizadas a su nombre canónico)**")
                ents["canonico"] = [etiqueta(S, c) if t in ("empleado", "proyecto", "certificacion", "curso") else c for t, c in zip(ents.tipo, ents.canonico)]
                st.dataframe(ents[["tipo", "canonico", "confianza"]], hide_index=True, width="stretch", height=180)
            else:
                st.caption("No se detectaron entidades del catálogo.")
            st.markdown("**Relaciones propuestas** (estado «propuesto»: confirme las que sean correctas)")
            if props:
                dfp = pd.DataFrame([dict(aceptar=True, origen=etiqueta(S, p["origen"]), relacion=p["relacion"], destino=etiqueta(S, p["destino"])) for p in props])
                ed = st.data_editor(dfp, hide_index=True, width="stretch", disabled=["origen", "relacion", "destino"], key=f"ed_{a['titulo']}_{cat_final}")
                aceptadas = [p for p, ok in zip(props, ed["aceptar"]) if ok]
            else:
                st.caption("Este tipo de documento no genera relaciones nuevas; se registrará y quedará disponible para búsqueda, asistente y recomendación.")
                aceptadas = []
            if st.button("✅ Confirmar y agregar al repositorio y al grafo"):
                try:
                    nuevo = S.registrar_documento(a["titulo"], a["texto"], cat_final, aceptadas, corregida_por_usuario=(cat_final != a["categoria"]))
                    st.success(f"Documento {nuevo['id']} registrado como «{cat_final}». Relaciones aplicadas: {nuevo.get('relaciones_aplicadas', 0)}. "
                               f"Ya se puede consultar desde el asistente, las recomendaciones y el grafo.")
                    st.session_state["ultimo_doc"] = nuevo["id"]
                    del st.session_state["analisis"]
                except Exception as ex:  # noqa: BLE001
                    st.error(f"No se pudo registrar el documento ({type(ex).__name__}: {ex}).")
    st.divider()
    st.subheader("Repositorio de documentos")
    cats = st.multiselect("Filtrar por categoría", CATEGORIAS_DOC, default=[])
    docs = pd.DataFrame([dict(id=d["id"], titulo=d["titulo"], categoria=d["categoria"], palabras=len(d["texto"].split())) for d in S.kb.docs])
    if cats:
        docs = docs[docs.categoria.isin(cats)]
    st.dataframe(docs, hide_index=True, width="stretch", height=240)
    sel = st.selectbox("Ver contenido de un documento", [""] + list(docs.id), format_func=lambda i: "(seleccione)" if not i else f"{i} · {S.kb.documento(i)['titulo']}")
    if sel:
        st.text(S.kb.documento(sel)["texto"])

# ================================================================== RECOMENDACIONES
with tabs[3]:
    st.header("Recomendaciones")
    t1, t2 = st.tabs(["📑 Documentos relacionados", "🎓 Capacitación personalizada"])
    with t1:
        st.caption("Similitud coseno entre embeddings del contenido (70 %) y coincidencia de conceptos del grafo (30 %). Se muestra la información usada para cada recomendación.")
        opciones = {d["id"]: f"{d['id']} · {d['titulo']}" for d in S.kb.docs}
        pref = st.session_state.get("ultimo_doc") or S.kb.doc_de(incidente="INC-0457")
        did = st.selectbox("Documento base", list(opciones), index=list(opciones).index(pref) if pref in opciones else 0, format_func=lambda i: opciones[i])
        kdoc = st.slider("Cantidad", 3, 10, 5, key="k_docs")
        rel = S.recomendador.relacionados(did, k=kdoc)
        st.dataframe(pd.DataFrame([dict(Documento=f"{r['doc_id']} · {r['titulo']}", Categoría=r["categoria"], Coseno=r["similitud_coseno"],
                                        Conceptos=r["jaccard_conceptos"], Puntaje=r["puntaje"], Compartidos=", ".join(r["conceptos_compartidos"]) or "—",
                                        Motivo=r["motivo"]) for r in rel]), hide_index=True, width="stretch")
    with t2:
        st.caption("Sugiere cursos según las competencias actuales y el objetivo profesional; explica el porqué de cada recomendación.")
        modo = st.radio("Perfil", ["Describir un perfil", "Empleado del directorio"], horizontal=True)
        if modo == "Describir un perfil":
            desc = st.text_area("Descripción", value="Daniel conoce Python básico y SQL, y quiere especializarse en análisis de datos", height=80)
            partes = re.split(r"\b(quiere|desea|busca|le interesa|interesad[oa]|especializarse|aprender)\b", desc, maxsplit=1, flags=re.I)
            actuales_txt, objetivo_txt = partes[0], "".join(partes[1:]) if len(partes) > 1 else ""
            hab = {}
            for e in S.kb.extractor.extraer(actuales_txt):
                from src.nlp import habilidad_de
                h = habilidad_de(e)
                if h:
                    hab[h] = 2 if re.search(r"b[aá]sic", actuales_txt, re.I) else 3
            objetivos = S.recomendador.objetivo_desde_texto(objetivo_txt)
            completados = set()
        else:
            emp = st.selectbox("Empleado", list(S.kb.empleados.id), format_func=lambda i: f"{S.kb.nombre_emp[i]} · {S.kb.empleados[S.kb.empleados.id == i].puesto.iloc[0]}")
            perfil = S.recomendador.perfil_de(emp)
            hab, completados = perfil["habilidades"], perfil["completados"]
            objetivo_txt = st.text_input("Objetivo profesional", value="quiere aprender machine learning")
            objetivos = S.recomendador.objetivo_desde_texto(objetivo_txt)
        st.markdown(f"**Habilidades actuales:** {', '.join(f'{h} (nivel {n})' for h, n in hab.items()) or 'ninguna detectada'}  \n**Objetivo detectado:** {', '.join(objetivos) or 'ninguno'}")
        if not objetivos:
            st.warning("No se detectó un objetivo del catálogo. Mencione una habilidad (por ejemplo: «quiere aprender análisis de datos»).")
        else:
            recs = S.recomendador.capacitacion(hab, objetivos, completados, k=6)
            if not recs:
                st.info("No hay cursos del catálogo que cumplan el objetivo con los prerrequisitos actuales.")
            for i, r in enumerate(recs, 1):
                st.markdown(f"**{i}. {r['curso']}** — {r['nivel']}, {r['horas']} h · relevancia {r['puntaje']:.2f}")
                st.caption(f"Por qué: {r['por_que']}.")

# ================================================================== ASISTENTE
with tabs[4]:
    st.header("Asistente de conocimiento (RAG)")
    st.caption("Responde con información recuperada de los documentos y del grafo, cita la fuente y se abstiene cuando no hay información suficiente.")
    ejq = ["¿Cuál es el procedimiento para solicitar un especialista?", "¿Quién resolvió el incidente INC-0457 y cómo?",
           "¿Qué certificaciones tiene Ana Torres?", "¿Quién tiene experiencia en Kubernetes?"]
    cols = st.columns(len(ejq))
    for i, q in enumerate(ejq):
        if cols[i].button(q, key=f"qa{i}", width="stretch"):
            st.session_state["pregunta_pendiente"] = q
    hist = st.session_state.setdefault("chat", [])
    entrada = st.chat_input("Escriba su pregunta…")
    entrada = entrada or st.session_state.pop("pregunta_pendiente", None)
    if entrada:
        try:
            hist.append(("user", entrada, None))
            hist.append(("assistant", None, S.asistente.responder(entrada)))
        except Exception as ex:  # noqa: BLE001
            hist.append(("assistant", f"Ocurrió un error al procesar la pregunta ({type(ex).__name__}).", None))
    for rol, txt, r in hist:
        with st.chat_message(rol):
            if rol == "user":
                st.write(txt)
            elif r is None:
                st.error(txt)
            else:
                if r["suficiente"]:
                    st.markdown(r["respuesta"])
                    st.markdown(chip("fundamentada" if r["fundamentada"] else "sin cita", "ef-ok" if r["fundamentada"] else "ef-warn") +
                                "".join(chip(f, "ef-info") for f in r["fuentes"]), unsafe_allow_html=True)
                else:
                    st.warning(r["respuesta"])
                    if r["fuentes"]:
                        st.markdown("".join(chip(f, "ef-info") for f in r["fuentes"]), unsafe_allow_html=True)
                    elif r.get("terminos_desconocidos"):
                        st.caption("Términos sin respaldo en los documentos: " + ", ".join(r["terminos_desconocidos"]))
                if r.get("aviso"):
                    st.caption(r["aviso"])
                with st.expander("Fuentes y contexto recuperado"):
                    st.caption(f"Mejor puntaje de recuperación: {r['mejor_puntaje']} · soporte de términos: {r['cobertura_terminos']}")
                    for h in r["hechos"][:4]:
                        st.markdown(f"• **Grafo** [{h['fuente']}]: {h['texto']}")
                    for f in r["fragmentos"][:3]:
                        st.markdown(f"• **{f['doc_id']} · {f['titulo']}** (puntaje {f['puntaje']:.2f}): {f['fragmento'][:260]}…")
    if hist and st.button("Limpiar conversación"):
        st.session_state["chat"] = []
        st.rerun()

# ================================================================== GRAFO
with tabs[5]:
    st.header("Grafo de conocimiento")
    st.caption("Arrastre los nodos, acerque con la rueda, pase el cursor para ver propiedades y use los botones de navegación.")
    leyenda = " ".join(f'<span class="ef-chip" style="background:{c};color:{"#000" if t in ("Tecnología", "Documento") else "#fff"}">{t}</span>' for t, c in COLORES.items())
    st.markdown(leyenda, unsafe_allow_html=True)
    modo_g = st.radio("Vista", ["Entorno de un nodo", "Por tipos de nodo"], horizontal=True)
    if modo_g == "Entorno de un nodo":
        c1, c2, c3 = st.columns([2, 1, 1])
        busca = c1.text_input("Buscar nodo (nombre o código)", value="Ana Torres")
        radio = c2.radio("Distancia", [1, 2], horizontal=True)
        incl_docs = c3.checkbox("Incluir documentos", value=False)
        cand = S.kb.buscar_nodo(busca)
        if not cand:
            st.warning("No se encontró ningún nodo con ese nombre. Pruebe con «Python», «INC-0457» o «Portal Bancario».")
        else:
            nodo = st.selectbox("Nodo", cand[:25], format_func=lambda n: f"{S.kb.G.nodes[n]['label']} ({S.kb.G.nodes[n]['tipo']})")
            tipos = {t for t in COLORES} - ({"Documento"} if not incl_docs else set())
            sub = S.kb.subgrafo(nodo, radio, tipos=tipos, max_nodos=130)
            st.caption(f"{sub.number_of_nodes()} nodos · {sub.number_of_edges()} relaciones")
            mostrar_grafo(S, sub, 640, {nodo})
    else:
        tipos_sel = st.multiselect("Tipos de nodo", list(COLORES), default=["Empleado", "Habilidad", "Certificación", "Proyecto", "Incidente", "Tecnología"])
        sub = S.kb.G.subgraph([n for n, d in S.kb.G.nodes(data=True) if d["tipo"] in tipos_sel]).copy()
        if sub.number_of_nodes() > 220:
            st.warning("Hay demasiados nodos para mostrar con claridad; reduzca los tipos seleccionados.")
        else:
            st.caption(f"{sub.number_of_nodes()} nodos · {sub.number_of_edges()} relaciones")
            mostrar_grafo(S, sub, 680)
    with st.expander("Estadísticas del grafo"):
        r = S.kb.resumen()
        a, b = st.columns(2)
        a.dataframe(pd.Series(r["tipos_nodo"], name="nodos").to_frame(), width="stretch")
        b.dataframe(pd.Series(r["tipos_relacion"], name="relaciones").to_frame(), width="stretch")

# ================================================================== EVALUACIÓN
with tabs[6]:
    st.header("Evaluación del prototipo")
    mf = RAIZ / "reports" / "metrics.json"
    if st.button("Ejecutar evaluación ahora"):
        with st.spinner("Ejecutando pruebas…"):
            from src.evaluation import ejecutar
            ejecutar()
        st.success("Evaluación completada.")
    if not mf.exists():
        st.info("Aún no hay resultados. Pulse «Ejecutar evaluación ahora».")
    else:
        m = json.load(open(mf, encoding="utf-8"))
        c = m["clasificacion"]
        st.subheader("Clasificación de documentos")
        k = st.columns(4)
        k[0].metric("Precision (macro)", f"{c['cv_precision_macro']:.3f}")
        k[1].metric("Recall (macro)", f"{c['cv_recall_macro']:.3f}")
        k[2].metric("F1-Score (macro)", f"{c['cv_f1_macro']:.3f}")
        k[3].metric("Accuracy", f"{c['cv_accuracy']:.3f}")
        st.caption(f"Validación cruzada estratificada de 5 particiones sobre {c['n_documentos']} documentos simulados.")
        a, b = st.columns(2)
        if (RAIZ / "figures" / "matriz_confusion.png").exists():
            a.image(str(RAIZ / "figures" / "matriz_confusion.png"))
        b.dataframe(pd.read_csv(RAIZ / "reports" / "clasificacion_reporte.csv", index_col=0).round(3), width="stretch")
        st.subheader("Recuperación: palabras clave vs. semántica")
        a, b = st.columns(2)
        if (RAIZ / "figures" / "comparacion_busqueda.png").exists():
            a.image(str(RAIZ / "figures" / "comparacion_busqueda.png"))
        b.dataframe(pd.read_csv(RAIZ / "reports" / "recuperacion_resumen.csv", index_col=0).round(3), width="stretch")
        st.subheader("Especialistas, recomendación y asistente")
        e, r, ch = m["expertos"], m["recomendacion"], m["chatbot"]
        k = st.columns(4)
        k[0].metric("Especialistas P@3", f"{e['P_at_3']:.2f}")
        k[1].metric("Tiempo de consulta", f"{e['tiempo_ms_promedio']} ms")
        k[2].metric("Docs relacionados P@3", f"{r['docs_P_at_3']:.2f}")
        k[3].metric("Asistente: respuestas correctas", f"{ch['porcentaje_respuestas_correctas']:.0%}")
        st.subheader("Tabla de pruebas")
        st.dataframe(pd.read_csv(RAIZ / "reports" / "tabla_pruebas.csv"), hide_index=True, width="stretch")
