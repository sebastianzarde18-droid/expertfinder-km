"""Generador de datos simulados de TalentTech Consulting.

Produce, de forma determinista (semilla fija):
  data/empleados.csv, habilidades_empleado.csv, certificaciones_empleado.csv,
  capacitaciones_empleado.csv, proyectos.csv, participaciones.csv, incidentes.csv,
  documentos.json y ground_truth.json (verdad de referencia usada solo para evaluar).

Todos los datos son ficticios; no contienen información personal real.
"""
from __future__ import annotations
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .ontology import (CERTIFICACIONES, CURSOS, DEPARTAMENTOS, HABILIDADES,
                       TECNOLOGIAS, FECHA_REFERENCIA)

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (nombre, arquetipo, departamento)
EMPLEADOS = [
    ("Ana Torres", "dev_python", 0), ("Luis Pérez", "dba", 0), ("Valeria Montoya", "analista", 1),
    ("Diego Herrera", "cientifico", 1), ("Camila Rojas", "fullstack", 0), ("Javier Salinas", "devops", 2),
    ("Fernanda Cárdenas", "lider", 4), ("Ricardo Vela", "seguridad", 3), ("Paola Guerrero", "qa", 3),
    ("Andrés Camacho", "dev_java", 0), ("Mariana Olvera", "ux", 0), ("Sergio Beltrán", "redes", 2),
    ("Daniela Ibarra", "soporte", 2), ("Héctor Quiroz", "cloud", 2), ("Lucía Navarro", "analista", 1),
    ("Emilio Cantú", "dev_python", 0), ("Regina Treviño", "lider", 4), ("Gabriel Leal", "dba", 1),
    ("Sofía Barrera", "fullstack", 0), ("Rodrigo Elizondo", "dev_java", 0), ("Karla Mendoza", "qa", 3),
    ("Iván Cavazos", "devops", 2), ("Natalia Fuentes", "cientifico", 1), ("Tomás Ochoa", "soporte", 2),
    ("Jimena Paredes", "consultora", 4), ("Mauricio Solís", "seguridad", 3), ("Alejandra Rangel", "dev_python", 0),
    ("Pablo Villarreal", "redes", 2), ("Renata Aguirre", "ux", 0), ("Óscar Maldonado", "fullstack", 0),
]

# arquetipo: puesto, habilidades principales, secundarias, tecnologías, certificaciones, cursos
ARQ = {
    "dev_python": ("Desarrollador/a Python", ["Python"], ["Bases de datos", "Pruebas de software"],
                   ["Django", "FastAPI", "PostgreSQL", "pandas"], ["C01"], ["K04", "K05", "K01"]),
    "dba": ("Especialista en bases de datos", ["Bases de datos"], ["Python"],
            ["PostgreSQL", "Oracle Database", "MySQL"], ["C02"], ["K02", "K03", "K01"]),
    "analista": ("Analista de datos", ["Análisis de datos", "Bases de datos"], ["Python"],
                 ["Power BI", "Tableau", "pandas", "PostgreSQL"], ["C10", "C11"], ["K06", "K09", "K24"]),
    "cientifico": ("Científico/a de datos", ["Machine Learning", "Python"], ["Análisis de datos", "Bases de datos"],
                   ["scikit-learn", "TensorFlow", "pandas"], ["C15"], ["K07", "K24", "K06"]),
    "fullstack": ("Desarrollador/a Full Stack", ["JavaScript"], ["Bases de datos", "Diseño UX/UI"],
                  ["React", "Node.js", "MongoDB", "MySQL"], [], ["K20", "K21", "K01"]),
    "devops": ("Ingeniero/a DevOps", ["DevOps"], ["Computación en la nube", "Redes"],
               ["Docker", "Kubernetes", "Jenkins", "AWS"], ["C12", "C13"], ["K10", "K11", "K12"]),
    "lider": ("Líder de proyecto", ["Gestión de proyectos", "Metodologías ágiles"], [],
              ["Jira"], ["C07", "C08"], ["K16", "K17"]),
    "seguridad": ("Analista de ciberseguridad", ["Ciberseguridad", "Redes"], [],
                  ["Splunk", "Wireshark"], ["C05"], ["K14", "K15"]),
    "qa": ("Ingeniero/a de calidad (QA)", ["Pruebas de software", "Metodologías ágiles"], [],
           ["Selenium", "Jira"], ["C09", "C08"], ["K18", "K16"]),
    "dev_java": ("Desarrollador/a Java", ["Java"], ["Bases de datos"],
                 ["Spring Boot", "Oracle Database", "PostgreSQL"], ["C14"], ["K22", "K01"]),
    "ux": ("Diseñador/a UX/UI", ["Diseño UX/UI"], ["JavaScript"],
           ["Figma", "React"], ["C17"], ["K19", "K20"]),
    "redes": ("Ingeniero/a de redes", ["Redes"], ["Ciberseguridad"],
              ["Cisco IOS", "Wireshark"], ["C06"], ["K15", "K14"]),
    "soporte": ("Analista de soporte técnico", ["Soporte técnico"], ["Redes", "Bases de datos"],
                ["Wireshark", "MySQL"], ["C16"], ["K23", "K15", "K01"]),
    "cloud": ("Arquitecto/a cloud", ["Computación en la nube", "DevOps"], ["Redes"],
              ["AWS", "Azure", "Kubernetes", "Docker"], ["C03", "C04"], ["K12", "K13", "K10"]),
    "consultora": ("Consultora de negocio", ["Gestión de proyectos", "Análisis de datos"], ["Metodologías ágiles"],
                   ["Jira", "Power BI"], ["C07", "C10"], ["K17", "K09", "K24"]),
}

PROYECTOS = [
    # id, nombre, cliente, sector, tecnologías, inicio, meses, estado
    ("P01", "Portal Bancario", "Banco Norte", "financiero", ["PostgreSQL", "Django", "Docker"], date(2024, 9, 2), 8, "finalizado"),
    ("P02", "Plataforma E-commerce Sol", "Retail Sol", "comercio", ["React", "Node.js", "MongoDB", "AWS"], date(2024, 11, 4), 7, "finalizado"),
    ("P03", "Tablero Analítico de Ventas", "Grupo Alfa", "manufactura", ["Power BI", "PostgreSQL", "pandas"], date(2025, 1, 13), 5, "finalizado"),
    ("P04", "Migración a la Nube", "Logística Rápida", "logística", ["AWS", "Docker", "Kubernetes", "Jenkins"], date(2025, 2, 3), 9, "finalizado"),
    ("P05", "Sistema de Inventarios", "Almacenes del Pacífico", "logística", ["Spring Boot", "Oracle Database"], date(2025, 3, 10), 8, "finalizado"),
    ("P06", "App de Seguros", "Seguros Horizonte", "financiero", ["React", "Node.js", "MySQL", "Figma"], date(2025, 4, 7), 6, "finalizado"),
    ("P07", "Sistema de Citas Hospitalarias", "Hospital Central", "salud", ["Django", "PostgreSQL", "Selenium"], date(2025, 5, 5), 7, "finalizado"),
    ("P08", "Modelo de Predicción de Demanda", "Grupo Alfa", "manufactura", ["pandas", "scikit-learn", "TensorFlow"], date(2025, 6, 16), 6, "finalizado"),
    ("P09", "Monitoreo de Infraestructura", "Telecom Norte", "telecomunicaciones", ["Splunk", "Wireshark", "Cisco IOS"], date(2025, 8, 4), 5, "finalizado"),
    ("P10", "Portal de Autoservicio de Clientes", "Banco Norte", "financiero", ["FastAPI", "PostgreSQL", "React", "Azure"], date(2025, 9, 1), 8, "en curso"),
    ("P11", "Modernización de Expedientes", "Secretaría Digital", "gobierno", ["Spring Boot", "PostgreSQL", "Jira"], date(2025, 10, 6), 9, "en curso"),
    ("P12", "Plataforma de Capacitación en Línea", "Universidad Regio", "educación", ["Django", "MySQL", "Figma"], date(2025, 11, 3), 6, "en curso"),
]

ROL_POR_ARQ = {
    "dev_python": "desarrollador backend", "dba": "especialista de datos", "analista": "analista de datos",
    "cientifico": "científico de datos", "fullstack": "desarrollador full stack", "devops": "ingeniero DevOps",
    "lider": "líder de proyecto", "seguridad": "analista de seguridad", "qa": "ingeniero de calidad",
    "dev_java": "desarrollador backend", "ux": "diseñador de interfaces", "redes": "ingeniero de redes",
    "soporte": "analista de soporte", "cloud": "arquitecto cloud", "consultora": "consultora funcional",
}

# tecnología -> [(síntoma, causa, solución, horas)]
INCIDENTES_PLANTILLA = {
    "PostgreSQL": [
        ("Caída de BD: la aplicación no responde y se agotan las conexiones", "saturación de conexiones",
         "reconfiguración del pool de conexiones y ajuste de índices", 3),
        ("Consultas lentas en horario pico", "consultas sin índice y bloqueos prolongados",
         "creación de índices y reescritura de consultas", 4),
        ("Disco al 100% en el servidor de base de datos", "acumulación de archivos WAL", "limpieza de WAL y configuración de archivado", 2),
    ],
    "Oracle Database": [
        ("Error ORA-01653 al insertar registros", "tablespace lleno", "ampliación del tablespace y monitoreo de espacio", 2),
        ("Degradación de rendimiento en reportes nocturnos", "estadísticas desactualizadas", "recolección de estadísticas y plan de ejecución fijo", 3),
    ],
    "MySQL": [
        ("Transacciones fallan con deadlock", "orden inconsistente de actualizaciones", "reordenamiento de transacciones y reintentos controlados", 5),
    ],
    "MongoDB": [
        ("Latencia alta en consultas de catálogo", "falta de índices compuestos", "creación de índices compuestos", 3),
    ],
    "Docker": [
        ("Contenedor se reinicia continuamente", "memoria insuficiente asignada al contenedor", "ajuste de límites de memoria y reconstrucción de la imagen", 2),
    ],
    "Kubernetes": [
        ("Pods en estado CrashLoopBackOff tras el despliegue", "variable de entorno faltante en el manifiesto", "corrección del manifiesto y nuevo despliegue", 3),
    ],
    "AWS": [
        ("Usuarios sin acceso a archivos en el almacenamiento", "permisos incorrectos en el bucket S3", "corrección de la política de acceso del bucket", 2),
    ],
    "Azure": [
        ("Fallo de autenticación en el portal", "secreto de aplicación expirado", "rotación del secreto y alerta de vencimiento", 2),
    ],
    "Node.js": [
        ("Servicio se detiene por consumo de memoria", "fuga de memoria en un manejador de eventos", "corrección del manejador y reinicio programado", 6),
    ],
    "Cisco IOS": [
        ("Pérdida de conectividad entre sucursales", "configuración errónea de VLAN en el enlace troncal", "corrección de la VLAN y validación con captura de tráfico", 3),
    ],
    "Splunk": [
        ("Alertas de múltiples intentos de acceso fallidos", "intento de acceso por fuerza bruta", "bloqueo de direcciones, rotación de credenciales y regla de alerta", 4),
    ],
    "Jenkins": [
        ("Pipeline de despliegue falla en la etapa de pruebas", "dependencia con versión incompatible", "fijado de versiones y limpieza de caché del agente", 2),
    ],
    "Spring Boot": [
        ("Servicio devuelve error 500 en el módulo de inventario", "excepción no controlada al consultar datos nulos", "validación de datos y manejo de excepciones", 3),
    ],
    "Django": [
        ("Migración falla al desplegar nueva versión", "conflicto entre migraciones paralelas", "fusión de migraciones y despliegue controlado", 2),
    ],
}

# tecnología -> habilidad(es) de quien suele resolver incidentes
ARQ_RESOLUTOR = {
    "PostgreSQL": ["dba", "dev_python"], "Oracle Database": ["dba"], "MySQL": ["dba", "soporte"], "MongoDB": ["dba", "fullstack"],
    "Docker": ["devops", "cloud"], "Kubernetes": ["devops", "cloud"], "AWS": ["cloud", "devops"], "Azure": ["cloud"],
    "Node.js": ["fullstack"], "Cisco IOS": ["redes"], "Splunk": ["seguridad"], "Jenkins": ["devops"],
    "Spring Boot": ["dev_java"], "Django": ["dev_python"],
}

ALIAS_RUIDO = {"PostgreSQL": ["Postgres"], "Bases de datos": ["BD", "bases de datos relacionales"], "Python": ["Python 3", "py"],
               "Kubernetes": ["k8s"], "JavaScript": ["JS"], "Machine Learning": ["aprendizaje automático"],
               "Computación en la nube": ["nube", "cloud"], "Pruebas de software": ["QA", "testing"]}


def _fecha(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _sumar_anios(d: date, anios: int) -> date:
    try:
        return d.replace(year=d.year + anios)
    except ValueError:
        return d.replace(year=d.year + anios, day=28)


def generar(seed: int = SEED, out_dir: Path = DATA_DIR) -> dict:
    rnd = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Empleados, habilidades, certificaciones, capacitaciones ----------------
    emp_rows, hab_rows, cert_rows, cap_rows = [], [], [], []
    verdad = {}  # empleado -> habilidades "reales"
    emp_tecn = {}
    for i, (nombre, arq, dep) in enumerate(EMPLEADOS, start=1):
        eid = f"EMP-{i:03d}"
        puesto, prim, sec, tecs, certs, cursos = ARQ[arq]
        ingreso = date(rnd.randint(2018, 2024), rnd.randint(1, 12), rnd.randint(1, 28))
        disp = rnd.choice(["disponible", "parcial", "parcial", "ocupado"])
        emp_rows.append(dict(id=eid, nombre=nombre, puesto=puesto, departamento=DEPARTAMENTOS[dep],
                             ingreso=ingreso.isoformat(), disponibilidad=disp, arquetipo=arq))
        habs = {h: rnd.randint(3, 5) for h in prim}
        for h in sec:
            if rnd.random() < 0.6:
                habs[h] = rnd.randint(2, 4)
        if nombre == "Ana Torres":
            habs = {"Python": 4, "Bases de datos": 3}
        if nombre == "Luis Pérez":
            habs = {"Bases de datos": 5}
        for h, nivel in habs.items():
            hab_rows.append(dict(empleado=eid, habilidad=h, nivel=nivel, anios=max(1, nivel + rnd.randint(-1, 3))))
        verdad[eid] = sorted(habs)
        k = 3 if arq in ("dev_python", "dba", "analista", "devops", "cloud") else 2
        emp_tecn[eid] = rnd.sample(tecs, min(k, len(tecs)))
        if nombre == "Ana Torres":
            emp_tecn[eid] = ["Django", "FastAPI", "PostgreSQL"]
        if nombre == "Luis Pérez":
            emp_tecn[eid] = ["PostgreSQL", "Oracle Database"]
        # certificaciones
        for cid in certs:
            if nombre in ("Ana Torres", "Luis Pérez") or rnd.random() < 0.75:
                nom, emisor, hab, vig = CERTIFICACIONES[cid]
                if nombre == "Ana Torres":
                    obt = date(2024, 5, 14)
                elif nombre == "Luis Pérez":
                    obt = date(2023, 3, 2)
                else:
                    obt = date(rnd.randint(2021, 2026), rnd.randint(1, 11), rnd.randint(1, 28))
                    if obt > FECHA_REFERENCIA:
                        obt = FECHA_REFERENCIA - timedelta(days=rnd.randint(30, 200))
                venc = _sumar_anios(obt, vig) if vig else None
                cert_rows.append(dict(empleado=eid, certificacion=cid, obtencion=obt.isoformat(),
                                      vigencia=venc.isoformat() if venc else "", folio=f"TT-{rnd.randint(100000, 999999)}"))
        # capacitaciones
        cur = list(cursos)
        if nombre == "Ana Torres":
            cur = ["K05"]
        if nombre == "Luis Pérez":
            cur = ["K02"]
        for kid in cur[: rnd.randint(1, len(cur))] if nombre not in ("Ana Torres", "Luis Pérez") else cur:
            cap_rows.append(dict(empleado=eid, curso=kid,
                                 fecha=(date(rnd.randint(2023, 2026), rnd.randint(1, 10), rnd.randint(1, 28))).isoformat(),
                                 calificacion=rnd.randint(82, 100)))
    empleados = pd.DataFrame(emp_rows)
    habilidades = pd.DataFrame(hab_rows)
    certs_emp = pd.DataFrame(cert_rows)
    caps_emp = pd.DataFrame(cap_rows)

    # ---------------- Proyectos y participaciones ----------------
    proy_rows, part_rows = [], []
    lideres = list(empleados[empleados.arquetipo.isin(["lider"])].id) + list(empleados[empleados.arquetipo == "consultora"].id)
    for j, (pid, nom, cli, sec, tecs, ini, meses, est) in enumerate(PROYECTOS):
        fin = ini + timedelta(days=30 * meses)
        proy_rows.append(dict(id=pid, nombre=nom, cliente=cli, sector=sec, tecnologias="|".join(tecs),
                              inicio=ini.isoformat(), fin=fin.isoformat(), estado=est))
        miembros = []
        lid = lideres[j % len(lideres)]
        miembros.append((lid, "líder de proyecto"))
        for _, e in empleados.iterrows():
            if e.arquetipo in ("lider", "consultora"):
                continue
            tecs_emp = set(emp_tecn[e.id])
            ov = len(tecs_emp & set(tecs))
            if ov and rnd.random() < (0.55 + 0.2 * ov):
                miembros.append((e.id, ROL_POR_ARQ[e.arquetipo]))
        # Aseguramos el caso del avance: Ana participó en Portal Bancario como desarrolladora
        if pid == "P01" and not any(m[0] == "EMP-001" for m in miembros):
            miembros.append(("EMP-001", "desarrollador backend"))
        for emp_id, rol in miembros:
            part_rows.append(dict(empleado=emp_id, proyecto=pid, rol=rol, inicio=ini.isoformat(), fin=fin.isoformat()))
    # Luis no participa como miembro en Portal Bancario (solo resolvió el incidente), como en el avance
    proyectos = pd.DataFrame(proy_rows)
    particip = pd.DataFrame(part_rows)
    particip = particip[~((particip.empleado == "EMP-002") & (particip.proyecto == "P01"))]

    # ---------------- Incidentes ----------------
    inc_rows = []
    num = 440
    usados = set()
    # Incidente semilla del avance
    inc_rows.append(dict(id="INC-0457", proyecto="P01", tecnologia="PostgreSQL", severidad="alta",
                         sintoma=INCIDENTES_PLANTILLA["PostgreSQL"][0][0], causa="saturación de conexiones",
                         solucion="reconfiguración y ajuste de índices", horas=3, fecha="2025-03-18", responsable="EMP-002"))
    usados.add(("P01", "PostgreSQL", 0))
    proy_info = {p[0]: p for p in PROYECTOS}
    intentos = 0
    while len(inc_rows) < 18 and intentos < 500:
        intentos += 1
        pid = rnd.choice([p[0] for p in PROYECTOS])
        tecs = [t for t in proy_info[pid][4] if t in INCIDENTES_PLANTILLA]
        if not tecs:
            continue
        t = rnd.choice(tecs)
        idx = rnd.randrange(len(INCIDENTES_PLANTILLA[t]))
        if (pid, t, idx) in usados:
            continue
        usados.add((pid, t, idx))
        sint, causa, sol, hrs = INCIDENTES_PLANTILLA[t][idx]
        cand = [e for e in empleados.itertuples() if e.arquetipo in ARQ_RESOLUTOR[t] and e.id != "EMP-002"]
        miembros_p = set(particip[particip.proyecto == pid].empleado)
        pref = [e for e in cand if e.id in miembros_p] or cand
        resp = rnd.choice(pref).id
        num += rnd.randint(3, 40)
        ini = proy_info[pid][5]
        f = ini + timedelta(days=rnd.randint(30, 30 * proy_info[pid][6] - 5))
        inc_rows.append(dict(id=f"INC-{num:04d}", proyecto=pid, tecnologia=t, severidad=rnd.choice(["alta", "media", "media", "baja"]),
                             sintoma=sint, causa=causa, solucion=sol, horas=hrs, fecha=f.isoformat(), responsable=resp))
    incidentes = pd.DataFrame(inc_rows)

    # ---------------- Documentos ----------------
    docs = _generar_documentos(rnd, empleados, habilidades, certs_emp, caps_emp, proyectos, particip, incidentes, emp_tecn)

    # ---------------- Guardado ----------------
    empleados.to_csv(out_dir / "empleados.csv", index=False)
    habilidades.to_csv(out_dir / "habilidades_empleado.csv", index=False)
    certs_emp.to_csv(out_dir / "certificaciones_empleado.csv", index=False)
    caps_emp.to_csv(out_dir / "capacitaciones_empleado.csv", index=False)
    proyectos.to_csv(out_dir / "proyectos.csv", index=False)
    particip.to_csv(out_dir / "participaciones.csv", index=False)
    incidentes.to_csv(out_dir / "incidentes.csv", index=False)
    pd.DataFrame([dict(id=k, tecnologia=k, habilidad=v[0], tipo=v[1]) for k, v in TECNOLOGIAS.items()]).to_csv(
        out_dir / "tecnologias.csv", index=False)
    with open(out_dir / "documentos.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=1)
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump({"habilidades_reales": verdad, "tecnologias_empleado": emp_tecn}, f, ensure_ascii=False, indent=1)
    return dict(empleados=len(empleados), documentos=len(docs), incidentes=len(incidentes), proyectos=len(proyectos))


# ===================================================================================
#                                  GENERACIÓN DE DOCUMENTOS
# ===================================================================================
def _alias(rnd, t, p=0.3):
    return rnd.choice(ALIAS_RUIDO[t]) if t in ALIAS_RUIDO and rnd.random() < p else t


def _temas_de(tecnologias=(), habilidades=()):
    """Temas (habilidades canónicas) de un documento: sirve como verdad para evaluar la recuperación."""
    out = set(habilidades)
    for t in tecnologias:
        if t in TECNOLOGIAS:
            out.add(TECNOLOGIAS[t][0])
    return sorted(out)


def _generar_documentos(rnd, emp, hab, certs_emp, caps_emp, proy, part, inc, emp_tecn):
    docs = []

    def add(cat, titulo, texto, **meta):
        docs.append(dict(id=f"D{len(docs) + 1:03d}", categoria=cat, titulo=titulo, texto=texto.strip(), **meta))

    nombre_de = dict(zip(emp.id, emp.nombre))
    puesto_de = dict(zip(emp.id, emp.puesto))
    proy_nom = dict(zip(proy.id, proy.nombre))

    # ---- CVs ----
    for e in emp.itertuples():
        hs = hab[hab.empleado == e.id]
        hlist = [f"{_alias(rnd, h.habilidad)} (nivel {h.nivel}/5, {h.anios} años)" for h in hs.itertuples()]
        tlist = [_alias(rnd, t, 0.4) for t in emp_tecn[e.id]]
        cl = [CERTIFICACIONES[c.certificacion][0] for c in certs_emp[certs_emp.empleado == e.id].itertuples()]
        cu = [CURSOS[c.curso][0] for c in caps_emp[caps_emp.empleado == e.id].itertuples()]
        pj = [proy_nom[p.proyecto] + f" ({p.rol})" for p in part[part.empleado == e.id].itertuples()]
        cab = rnd.choice(["CURRICULUM VITAE\n", "Perfil profesional\n", "", "Hoja de vida\n"])
        if rnd.random() < 0.5:
            t = (f"{cab}Nombre: {e.nombre}. Puesto actual: {e.puesto} en {e.departamento}. Ingreso a TalentTech Consulting en {e.ingreso[:4]}.\n"
                 f"Resumen: profesional con experiencia en {', '.join(hlist)}.\n"
                 f"Tecnologías que utilizo: {', '.join(tlist)}.\n"
                 f"Certificaciones: {', '.join(cl) if cl else 'ninguna registrada'}.\n"
                 f"Formación continua: {', '.join(cu) if cu else 'sin cursos registrados'}.\n"
                 f"Proyectos recientes: {', '.join(pj[:4]) if pj else 'sin proyectos asignados'}.\n"
                 f"Disponibilidad: {e.disponibilidad}.")
        else:
            t = (f"{cab}{e.nombre} | {e.puesto}\n"
                 f"Me desempeño en {e.departamento} desde {e.ingreso[:4]}. Mis áreas de dominio son {', '.join(hlist)}; "
                 f"trabajo con {', '.join(tlist)}. "
                 f"{'He obtenido ' + ', '.join(cl) + '. ' if cl else ''}"
                 f"{'Complete los cursos ' + ', '.join(cu) + '. ' if cu else ''}"
                 f"{'He participado en ' + ', '.join(pj[:4]) + '.' if pj else ''}")
        # ruido: un tercio de los CV incluye una frase típica de otro tipo de documento
        if rnd.random() < 0.3:
            t += rnd.choice([" Referencias: disponibles a solicitud.", " Atendí incidentes de producción en guardias rotativas.",
                             " Participé en el cierre de proyectos y en sus lecciones aprendidas."])
        add("CV o perfil profesional", f"CV de {e.nombre}", t, empleado=e.id,
            temas=_temas_de(emp_tecn[e.id], list(hs.habilidad)))

    # ---- Certificados y constancias ----
    for c in certs_emp.itertuples():
        nom, emisor, habil, vig = CERTIFICACIONES[c.certificacion]
        e = nombre_de[c.empleado]
        cab = rnd.choice(["CONSTANCIA DE CERTIFICACIÓN\n", "CERTIFICADO\n", "", "Constancia de aprobación\n"])
        v = f"Vigencia hasta: {_fecha(date.fromisoformat(c.vigencia))}." if c.vigencia else "Vigencia: sin fecha de vencimiento."
        t = (f"{cab}{emisor} hace constar que {e} aprobó el examen de la certificación «{nom}». "
             f"Fecha de obtención: {_fecha(date.fromisoformat(c.obtencion))}. {v} Folio {c.folio}. "
             f"La certificación avala conocimientos de {habil}.")
        if rnd.random() < 0.25:
            t += " Documento digital emitido por el organismo certificador."
        add("Certificado o constancia", f"Certificado {nom} – {e}", t, empleado=c.empleado, certificacion=c.certificacion,
            temas=[habil])
    for c in caps_emp.itertuples():
        nom, habs_, nivel, pre, hrs, desc = CURSOS[c.curso]
        e = nombre_de[c.empleado]
        cab = rnd.choice(["CONSTANCIA DE CURSO\n", "Constancia de participación\n", ""])
        t = (f"{cab}Se hace constar que {e} concluyó satisfactoriamente el curso «{nom}» con una duración de {hrs} horas "
             f"y calificación de {c.calificacion}/100, con fecha {_fecha(date.fromisoformat(c.fecha))}. "
             f"El curso desarrolla: {', '.join(habs_)}. {desc}.")
        add("Certificado o constancia", f"Constancia {nom} – {e}", t, empleado=c.empleado, curso=c.curso, temas=list(habs_))

    # ---- Informes de cierre ----
    for p in proy.itertuples():
        tecs = p.tecnologias.split("|")
        eq = part[part.proyecto == p.id]
        equipo = "; ".join(f"{nombre_de[x.empleado]} ({x.rol})" for x in eq.itertuples())
        incs = inc[inc.proyecto == p.id]
        il = ", ".join(incs.id) if len(incs) else "ninguno"
        cab = rnd.choice(["INFORME DE CIERRE DE PROYECTO\n", "Informe final del proyecto\n", "", "Cierre de proyecto y lecciones aprendidas\n"])
        t = (f"{cab}Proyecto: {p.nombre}. Cliente: {p.cliente} (sector {p.sector}). Periodo: {_fecha(date.fromisoformat(p.inicio))} al "
             f"{_fecha(date.fromisoformat(p.fin))}. Estado: {p.estado}.\n"
             f"Tecnologías empleadas: {', '.join(_alias(rnd, x, 0.25) for x in tecs)}.\n"
             f"Equipo: {equipo}.\n"
             f"Resultados: se entregaron los módulos comprometidos con el cliente {p.cliente}, cumpliendo los criterios de aceptación. "
             f"Incidentes relevantes: {il}.\n"
             f"Lecciones aprendidas: documentar decisiones de arquitectura desde el inicio y registrar las habilidades aplicadas por cada integrante.")
        if rnd.random() < 0.3:
            t += " Se recomienda actualizar los perfiles del equipo con las habilidades aplicadas."
        add("Informe de cierre de proyecto", f"Informe de cierre – {p.nombre}", t, proyecto=p.id, temas=_temas_de(tecs))

    # ---- Tickets ----
    for i in inc.itertuples():
        cab = rnd.choice(["TICKET DE INCIDENTE\n", "Ticket ", "Registro de incidente\n"])
        t = (f"{cab}{i.id}. Severidad: {i.severidad}. Proyecto afectado: {proy_nom[i.proyecto]}. Fecha: {_fecha(date.fromisoformat(i.fecha))}.\n"
             f"Síntoma: {i.sintoma}. Tecnología involucrada: {_alias(rnd, i.tecnologia, 0.3)}.\n"
             f"Causa raíz: {i.causa}.\n"
             f"Solución aplicada: {i.solucion}.\n"
             f"Responsable: {nombre_de[i.responsable]}. Tiempo de resolución: {i.horas} horas.")
        if rnd.random() < 0.25:
            t += " Estado del ticket: cerrado."
        add("Ticket de incidente", f"Ticket {i.id}", t, incidente=i.id, proyecto=i.proyecto, empleado=i.responsable,
            temas=_temas_de([i.tecnologia]))

    # ---- Minutas (vocabulario deliberadamente mezclado con otros tipos) ----
    for k in range(14):
        p = proy.sample(1, random_state=rnd.randint(0, 9999)).iloc[0]
        tecs = p.tecnologias.split("|")
        eq = part[part.proyecto == p.id]
        asist = [nombre_de[x] for x in eq.empleado.tolist()[:4]]
        f = date(2025, 1, 6) + timedelta(days=rnd.randint(0, 300))
        incs = inc[inc.proyecto == p.id]
        acuerdos = [
            f"Revisar el avance del módulo de {_alias(rnd, rnd.choice(tecs), 0.3)} antes del siguiente corte.",
            "Actualizar el tablero de seguimiento y los riesgos del proyecto.",
            f"Programar capacitación breve sobre {rnd.choice(tecs)} para el equipo.",
        ]
        if len(incs):
            acuerdos.append(f"Dar seguimiento a la causa raíz del incidente {incs.iloc[0].id} y documentar la solución.")
        if rnd.random() < 0.5:
            acuerdos.append("Renovar las certificaciones próximas a vencer de los integrantes.")
        cab = rnd.choice(["MINUTA DE REUNIÓN\n", "Minuta de seguimiento\n", "", "Acta de reunión\n"])
        t = (f"{cab}Proyecto: {p.nombre}. Fecha: {_fecha(f)}. Asistentes: {', '.join(asist)}.\n"
             f"Temas tratados: avance del proyecto con {p.cliente}, riesgos y asignación de tareas.\n"
             f"Acuerdos: " + " ".join(f"({n}) {a}" for n, a in enumerate(acuerdos, 1)))
        add("Minuta de reunión", f"Minuta {p.nombre} – {_fecha(f)}", t, proyecto=p.id, temas=_temas_de(tecs))

    # ---- Manuales y políticas (redactados manualmente; base para el chatbot) ----
    for titulo, texto, temas in MANUALES:
        add("Manual o política interna", titulo, texto, temas=temas)

    rnd.shuffle(docs)
    for n, d in enumerate(docs, start=1):  # reasigna ids tras mezclar
        d["id"] = f"D{n:03d}"
    return docs


MANUALES = [
    ("Procedimiento para solicitar un especialista", """PROCEDIMIENTO PARA SOLICITAR UN ESPECIALISTA
Objetivo: formar equipos de proyecto con personal que tenga evidencia comprobable de sus habilidades.
Paso 1. El líder de proyecto registra la solicitud en ExpertFinder indicando las habilidades requeridas, el nivel mínimo y las fechas.
Paso 2. El sistema muestra los especialistas ordenados por coincidencia y por solidez de la evidencia.
Paso 3. El líder revisa las evidencias de cada candidato (certificaciones vigentes, proyectos e incidentes) y selecciona hasta tres.
Paso 4. Recursos Humanos confirma la disponibilidad del colaborador en un plazo máximo de 2 días hábiles.
Paso 5. Gestión de Proyectos formaliza la asignación y notifica al colaborador y a su jefe directo.""", []),
    ("Procedimiento de atención y escalamiento de incidentes", """PROCEDIMIENTO DE ATENCIÓN Y ESCALAMIENTO DE INCIDENTES
Todo incidente se registra como ticket con código INC, severidad, proyecto afectado y síntoma.
Severidad alta: atención inmediata y escalamiento al líder técnico en menos de 30 minutos.
Severidad media: atención en un máximo de 4 horas hábiles. Severidad baja: atención en 2 días hábiles.
Antes de escalar, el analista consulta en ExpertFinder los incidentes anteriores sobre la misma tecnología para reutilizar la solución documentada.
Al cerrar el ticket se debe registrar causa raíz, solución aplicada, responsable y tiempo de resolución.""", ["Soporte técnico"]),
    ("Política de vigencia y renovación de certificaciones", """POLÍTICA DE VIGENCIA Y RENOVACIÓN DE CERTIFICACIONES
Las certificaciones con fecha de vencimiento se consideran evidencia vigente solo hasta esa fecha.
Una certificación vencida no cuenta como evidencia ni suma puntos en el ranking de especialistas.
El sistema genera un aviso 90 días antes del vencimiento para el colaborador y para Recursos Humanos.
La empresa apoya la renovación de certificaciones relevantes para los proyectos activos, previa solicitud del jefe directo.""", []),
    ("Guía de validación de habilidades propuestas", """GUÍA DE VALIDACIÓN DE HABILIDADES PROPUESTAS
Cuando el sistema extrae habilidades de un documento, estas quedan con estado «propuesto».
El colaborador o Recursos Humanos tienen 5 días hábiles para confirmar, corregir o rechazar cada propuesta.
Una habilidad solo pasa a «comprobada» si cuenta con al menos una evidencia vigente: certificación, proyecto, incidente o capacitación.
Las habilidades sin evidencia se muestran como «declaradas» y pesan menos en el ranking.""", []),
    ("Política de privacidad y protección de datos personales", """POLÍTICA DE PRIVACIDAD Y PROTECCIÓN DE DATOS PERSONALES
TalentTech Consulting trata los datos personales de sus colaboradores conforme a la LFPDPPP.
Cada colaborador otorga su consentimiento para que su perfil profesional se almacene y consulte en el directorio.
El acceso se controla por roles: Recursos Humanos y Dirección ven todos los datos; los líderes de proyecto ven perfiles profesionales; los colaboradores editan solo su propio perfil.
No se almacenan datos sensibles como salud, ideología o información financiera personal.
El colaborador puede solicitar la corrección o eliminación de su perfil en cualquier momento.""", []),
    ("Procedimiento de cierre de proyecto y lecciones aprendidas", """PROCEDIMIENTO DE CIERRE DE PROYECTO Y LECCIONES APRENDIDAS
El líder de proyecto elabora el informe de cierre dentro de los 10 días hábiles posteriores a la entrega final.
El informe incluye: cliente, periodo, equipo con roles, tecnologías empleadas, resultados, incidentes relevantes y lecciones aprendidas.
Una vez cargado el informe, el sistema propone habilidades aplicadas por cada integrante para que sean confirmadas.
Recursos Humanos revisa que los perfiles del equipo queden actualizados.""", []),
    ("Catálogo de niveles de habilidad", """CATÁLOGO DE NIVELES DE HABILIDAD
Nivel 1 Básico: conoce los conceptos y requiere supervisión constante.
Nivel 2 Elemental: realiza tareas sencillas con apoyo ocasional.
Nivel 3 Intermedio: trabaja de forma autónoma en tareas habituales.
Nivel 4 Avanzado: resuelve problemas complejos y orienta a otros.
Nivel 5 Experto: define estándares, diseña soluciones y es referente interno y externo.
El nivel declarado debe respaldarse con evidencias para considerarse comprobado.""", []),
    ("Procedimiento de registro de perfil e incorporación", """PROCEDIMIENTO DE REGISTRO DE PERFIL E INCORPORACIÓN
Recursos Humanos crea el perfil del nuevo colaborador durante su primera semana, con puesto, departamento y fecha de ingreso.
El colaborador carga su CV y sus certificados; el sistema clasifica cada documento y propone habilidades.
El colaborador confirma las propuestas y acepta el aviso de privacidad.
El jefe directo valida el nivel de las habilidades principales durante el primer mes.""", []),
    ("Política de capacitación", """POLÍTICA DE CAPACITACIÓN
Cada colaborador dispone de hasta 40 horas de capacitación al año financiadas por la empresa.
Las solicitudes se presentan a Recursos Humanos con la aprobación del jefe directo.
Se da prioridad a cursos que fortalezcan habilidades con pocos expertos internos.
Al concluir el curso, el colaborador carga su constancia para actualizar su perfil.""", []),
    ("Manual de uso del directorio ExpertFinder", """MANUAL DE USO DEL DIRECTORIO EXPERTFINDER
Buscar especialistas: escriba el requisito en lenguaje natural, por ejemplo «persona con experiencia en Python y bases de datos».
Subir documentos: cargue un CV, certificado, informe o ticket; el sistema propone su categoría y las entidades detectadas, que usted puede corregir.
Asistente: formule preguntas sobre procedimientos, incidentes o perfiles; las respuestas incluyen la fuente. Si no hay información suficiente, el asistente lo indicará.
Grafo: explore las relaciones entre empleados, habilidades, proyectos y documentos.""", []),
]


if __name__ == "__main__":
    print(generar())
