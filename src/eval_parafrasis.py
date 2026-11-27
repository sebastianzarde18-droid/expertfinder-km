"""Evaluación del asistente con preguntas PARAFRASEADAS (comparación extractivo vs LLM).

Las preguntas se redactaron con palabras distintas a las de los documentos. Hay dos conjuntos:
  * DEV  (15): se pueden usar para ajustar el sistema;
  * TEST (15): reservado; se mide una sola vez después de ajustar.
Cada pregunta: (texto, ¿tiene respuesta?, fragmentos esperados, título del documento fuente o None).

Uso:  python -m src.eval_parafrasis            # modo extractivo (sin claves)
      ANTHROPIC_API_KEY=... python -m src.eval_parafrasis --llm   # compara además el modo LLM
"""
from __future__ import annotations
import json, os, sys, time
import pandas as pd
from .sistema import Sistema
from .evaluation import REPORTES

DEV = [
    ("¿Qué pasos sigue un líder de proyecto para pedir a una persona con cierta habilidad?", True, ["Paso 1"], "Procedimiento para solicitar un especialista"),
    ("Si un certificado ya caducó, ¿sirve para posicionarse en la lista de candidatos?", True, ["no cuenta"], "Política de vigencia y renovación de certificaciones"),
    ("¿Cuántas horas al año puede dedicar un colaborador a cursos pagados por la empresa?", True, ["40 horas"], "Política de capacitación"),
    ("¿A cuántos minutos se debe avisar al líder técnico cuando una falla es grave?", True, ["30 minutos"], "Procedimiento de atención y escalamiento de incidentes"),
    ("¿Qué ley protege la información personal de los empleados?", True, ["LFPDPPP"], "Política de privacidad y protección de datos personales"),
    ("¿Cómo busco a un especialista dentro del sistema?", True, ["lenguaje natural"], "Manual de uso del directorio ExpertFinder"),
    ("¿Cuál fue el origen del problema en el ticket INC-0457?", True, ["saturación de conexiones"], "Ticket INC-0457"),
    ("¿Qué personas del equipo dominan Kubernetes?", True, ["Javier Salinas"], None),
    ("¿Cuántos días tiene el colaborador para ratificar las habilidades que sugirió el sistema?", True, ["5 días hábiles"], "Guía de validación de habilidades propuestas"),
    ("¿En qué lapso se redacta el reporte final de un proyecto?", True, ["10 días hábiles"], "Procedimiento de cierre de proyecto y lecciones aprendidas"),
    ("¿Cuánto gana Ana Torres al mes?", False, [], None),
    ("¿Cuántos días de vacaciones corresponden a cada empleado?", False, [], None),
    ("¿Dónde queda la oficina principal de la empresa?", False, [], None),
    ("¿Cómo se pide el reembolso de viáticos?", False, [], None),
    ("¿Quién es el director general de TalentTech?", False, [], None),
]
TEST = [
    ("¿Cuánto tarda Recursos Humanos en confirmar que el colaborador está libre?", True, ["2 días hábiles"], "Procedimiento para solicitar un especialista"),
    ("¿Con qué antelación me avisan que mi certificación está por expirar?", True, ["90 días"], "Política de vigencia y renovación de certificaciones"),
    ("¿Quién debe autorizar mi solicitud de curso?", True, ["jefe directo"], "Política de capacitación"),
    ("¿Qué datos hay que anotar al cerrar un ticket?", True, ["causa raíz"], "Procedimiento de atención y escalamiento de incidentes"),
    ("¿Cuál es el plazo de atención para un incidente de gravedad baja?", True, ["2 días hábiles"], "Procedimiento de atención y escalamiento de incidentes"),
    ("¿Quiénes pueden ver todos los datos guardados en el directorio?", True, ["Recursos Humanos"], "Política de privacidad y protección de datos personales"),
    ("¿Puedo pedir que borren mi perfil del sistema?", True, ["eliminación"], "Política de privacidad y protección de datos personales"),
    ("¿Qué hago para añadir mi currículum al directorio?", True, ["categoría"], "Manual de uso del directorio ExpertFinder"),
    ("¿Qué cursos de certificación ha completado Ana Torres?", True, ["PCAP"], "CV de Ana Torres"),
    ("¿Cómo se resolvió la falla INC-0505 de Kubernetes?", True, ["manifiesto"], "Ticket INC-0505"),
    ("¿Qué código de vestimenta se exige en la oficina?", False, [], None),
    ("¿Qué seguro médico ofrece la compañía a su personal?", False, [], None),
    ("¿Cómo se tramita la renuncia de un colaborador?", False, [], None),
    ("¿Cuál es la contraseña del servidor de producción?", False, [], None),
    ("¿Qué lenguaje de programación conviene aprender primero?", False, [], None),
]


def evaluar(S: Sistema, preguntas, usar_llm: bool, etiqueta: str) -> pd.DataFrame:
    a = S.asistente
    clave = a.api_key
    if not usar_llm:
        a.api_key = None  # fuerza el modo extractivo
    filas = []
    for q, responde, esperados, fuente in preguntas:
        t = time.perf_counter()
        r = a.responder(q)
        ms = (time.perf_counter() - t) * 1000
        txt = r["respuesta"]
        if responde:
            ok = r["suficiente"] and all(e.lower() in txt.lower() for e in esperados)
            if fuente:
                ids = {d["id"] for d in S.kb.docs if d["titulo"] == fuente}
                fund = bool(ids & set(r["fuentes"]))
            else:
                fund = bool(r["fuentes"])
            res = "correcta" if ok else ("falso rechazo" if not r["suficiente"] else "incorrecta")
        else:
            fund = None
            res = "rechazo correcto" if not r["suficiente"] else "respuesta inventada"
        filas.append(dict(conjunto=etiqueta, modo="LLM" if usar_llm else "extractivo", pregunta=q, tiene_respuesta=responde,
                          resultado=res, fundamentada=fund, usado_llm=r.get("usado_llm", False), aviso=r.get("aviso", ""), embeddings=S.repo.emb.nombre, ms=round(ms, 1),
                          respuesta=txt.replace("\n", " ")[:300]))
    a.api_key = clave
    return pd.DataFrame(filas)


def resumen(df: pd.DataFrame) -> dict:
    con, sin = df[df.tiene_respuesta], df[~df.tiene_respuesta]
    return dict(n=len(df), correctas=int((con.resultado == "correcta").sum()), de_con_respuesta=len(con),
                exactitud_con_respuesta=round(float((con.resultado == "correcta").mean()), 3),
                falsos_rechazos=int((df.resultado == "falso rechazo").sum()), incorrectas=int((df.resultado == "incorrecta").sum()),
                rechazos_correctos=int((sin.resultado == "rechazo correcto").sum()), de_sin_respuesta=len(sin),
                inventadas=int((df.resultado == "respuesta inventada").sum()),
                fundamentadas=int(con.fundamentada.astype(bool).sum()),
                ms=round(float(df.ms.mean()), 1))


if __name__ == "__main__":
    st = "--st" in sys.argv            # embeddings preentrenados (sentence-transformers); requiere descargar el modelo
    if st:
        os.environ["EF_EMBEDDINGS"] = "st"
    S = Sistema()
    print("Embeddings:", S.repo.emb.nombre)
    if "--llm" in sys.argv and not S.asistente.api_key:
        sys.exit("Falta ANTHROPIC_API_KEY: sin clave no se puede evaluar el modo LLM.")
    modos = [False] + ([True] if "--llm" in sys.argv else [])
    out, tablas = {}, []
    for llm in modos:
        for nom, conj in (("dev", DEV), ("test", TEST)):
            df = evaluar(S, conj, llm, nom)
            tablas.append(df)
            clave = f"{'llm' if llm else 'extractivo'}{'_st' if st else ''}_{nom}"
            out[clave] = resumen(df)
            if llm:
                out[clave]["respuestas_con_llm"] = int(df.usado_llm.sum())  # si es 0, el LLM no se usó (revisar clave / red)
                if df.aviso.astype(str).str.len().gt(0).any():
                    print("AVISO:", df.aviso[df.aviso.astype(str).str.len() > 0].iloc[0])
    suf = "_st" if st else ""
    if llm_pedido := ("--llm" in sys.argv):
        suf += "_llm"
    pd.concat(tablas).to_csv(REPORTES / f"parafrasis_detalle{suf}.csv", index=False)
    (REPORTES / f"parafrasis_resumen{suf}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
