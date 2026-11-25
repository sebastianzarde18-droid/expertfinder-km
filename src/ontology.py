"""Ontología de ExpertFinder: catálogos de clases, instancias, sinónimos y reglas.

Retoma la ontología del avance (Parte 3) y la amplía con: Documento, Categoría de
documento y Área de conocimiento. Los catálogos son datos simulados coherentes con
TalentTech Consulting.
"""
from __future__ import annotations
from datetime import date

# Fecha de referencia para evaluar la vigencia (regla R2). Se puede cambiar en la app.
FECHA_REFERENCIA = date(2026, 11, 23)

AREAS = {
    "Programación": "Lenguajes y desarrollo de software",
    "Datos e IA": "Gestión, análisis y modelado de datos",
    "Infraestructura": "Nube, redes, operación y despliegue",
    "Seguridad y calidad": "Ciberseguridad y aseguramiento de calidad",
    "Gestión y diseño": "Gestión de proyectos y diseño de producto",
}

# Habilidad canónica -> (área, sinónimos, descripción)
HABILIDADES = {
    "Python": ("Programación", ["py", "python3", "python 3", "programacion en python"],
               "Programación en Python para desarrollo de aplicaciones, scripts y automatización"),
    "Bases de datos": ("Datos e IA", ["bd", "bases datos", "base de datos", "sql", "db", "database", "gestion de datos"],
                       "Diseño, consulta, administración y optimización de bases de datos relacionales y no relacionales con SQL"),
    "JavaScript": ("Programación", ["js", "ecmascript", "desarrollo web", "frontend", "front end", "backend", "front-end"],
                   "Desarrollo web con JavaScript en cliente y servidor"),
    "Java": ("Programación", ["jvm", "java se", "java ee"],
             "Desarrollo de aplicaciones empresariales con Java"),
    "Análisis de datos": ("Datos e IA", ["analitica", "analisis de datos", "data analysis", "analytics", "bi", "inteligencia de negocios", "analizar datos"],
                          "Análisis, limpieza, visualización de datos y tableros de inteligencia de negocios"),
    "Machine Learning": ("Datos e IA", ["aprendizaje automatico", "ml", "inteligencia artificial", "ia", "aprendizaje de maquina", "ciencia de datos"],
                         "Modelos de aprendizaje automático, entrenamiento, evaluación y predicción"),
    "Computación en la nube": ("Infraestructura", ["nube", "cloud", "cloud computing"],
                               "Diseño y operación de soluciones en la nube pública"),
    "DevOps": ("Infraestructura", ["ci/cd", "integracion continua", "despliegue continuo", "infraestructura como codigo", "contenedores"],
               "Automatización de despliegues, integración continua y contenedores"),
    "Ciberseguridad": ("Seguridad y calidad", ["seguridad informatica", "seguridad", "hacking etico", "infosec"],
                       "Protección de sistemas, monitoreo de amenazas y respuesta a incidentes de seguridad"),
    "Redes": ("Infraestructura", ["networking", "redes de computadoras", "telecomunicaciones"],
              "Diseño, configuración y diagnóstico de redes de datos"),
    "Soporte técnico": ("Infraestructura", ["mesa de ayuda", "help desk", "soporte", "atencion de incidentes"],
                        "Atención de usuarios, diagnóstico y resolución de incidentes"),
    "Pruebas de software": ("Seguridad y calidad", ["qa", "testing", "calidad de software", "pruebas", "aseguramiento de calidad"],
                            "Diseño y ejecución de pruebas manuales y automatizadas"),
    "Gestión de proyectos": ("Gestión y diseño", ["project management", "administracion de proyectos", "pmo", "liderazgo de proyectos"],
                             "Planeación, seguimiento y liderazgo de proyectos tecnológicos"),
    "Metodologías ágiles": ("Gestión y diseño", ["scrum", "agile", "kanban", "agil", "metodologia agil"],
                            "Aplicación de Scrum, Kanban y prácticas ágiles"),
    "Diseño UX/UI": ("Gestión y diseño", ["ux", "ui", "diseno de interfaces", "experiencia de usuario", "diseno ux"],
                     "Investigación de usuarios y diseño de interfaces"),
}

# Tecnología -> (habilidad padre [es_un_tipo_de], tipo, sinónimos)
TECNOLOGIAS = {
    "PostgreSQL": ("Bases de datos", "base de datos", ["postgres", "pgsql", "postgresql"]),
    "MySQL": ("Bases de datos", "base de datos", ["mysql"]),
    "Oracle Database": ("Bases de datos", "base de datos", ["oracle db", "oracle"]),
    "MongoDB": ("Bases de datos", "base de datos", ["mongo", "mongodb"]),
    "Django": ("Python", "framework", ["django"]),
    "FastAPI": ("Python", "framework", ["fastapi"]),
    "pandas": ("Python", "biblioteca", ["pandas"]),
    "React": ("JavaScript", "framework", ["reactjs", "react.js", "react"]),
    "Node.js": ("JavaScript", "plataforma", ["nodejs", "node js", "node"]),
    "Spring Boot": ("Java", "framework", ["springboot", "spring"]),
    "Power BI": ("Análisis de datos", "herramienta", ["powerbi", "power bi"]),
    "Tableau": ("Análisis de datos", "herramienta", ["tableau"]),
    "scikit-learn": ("Machine Learning", "biblioteca", ["sklearn", "scikit learn", "scikit-learn"]),
    "TensorFlow": ("Machine Learning", "biblioteca", ["tensorflow", "keras"]),
    "AWS": ("Computación en la nube", "plataforma", ["amazon web services", "aws"]),
    "Azure": ("Computación en la nube", "plataforma", ["microsoft azure", "azure"]),
    "Docker": ("DevOps", "herramienta", ["docker", "contenedor docker"]),
    "Kubernetes": ("DevOps", "plataforma", ["k8s", "kubernetes"]),
    "Jenkins": ("DevOps", "herramienta", ["jenkins"]),
    "Wireshark": ("Redes", "herramienta", ["wireshark"]),
    "Cisco IOS": ("Redes", "plataforma", ["cisco", "cisco ios"]),
    "Splunk": ("Ciberseguridad", "herramienta", ["splunk", "siem"]),
    "Selenium": ("Pruebas de software", "herramienta", ["selenium"]),
    "Jira": ("Gestión de proyectos", "herramienta", ["jira"]),
    "Figma": ("Diseño UX/UI", "herramienta", ["figma"]),
}

# id -> (nombre, emisor, habilidad que avala, años de vigencia o None)
CERTIFICACIONES = {
    "C01": ("PCAP – Certified Associate in Python Programming", "Python Institute", "Python", None),
    "C02": ("Oracle Database SQL Certified Associate", "Oracle", "Bases de datos", None),
    "C03": ("AWS Certified Cloud Practitioner", "Amazon Web Services", "Computación en la nube", 3),
    "C04": ("Microsoft Azure Fundamentals (AZ-900)", "Microsoft", "Computación en la nube", None),
    "C05": ("CompTIA Security+", "CompTIA", "Ciberseguridad", 3),
    "C06": ("Cisco CCNA", "Cisco", "Redes", 3),
    "C07": ("PMP – Project Management Professional", "PMI", "Gestión de proyectos", 3),
    "C08": ("Professional Scrum Master I (PSM I)", "Scrum.org", "Metodologías ágiles", None),
    "C09": ("ISTQB Certified Tester Foundation Level", "ISTQB", "Pruebas de software", None),
    "C10": ("Google Data Analytics Certificate", "Google", "Análisis de datos", None),
    "C11": ("Microsoft Power BI Data Analyst (PL-300)", "Microsoft", "Análisis de datos", 1),
    "C12": ("Certified Kubernetes Administrator (CKA)", "CNCF", "DevOps", 2),
    "C13": ("Docker Certified Associate", "Docker Inc.", "DevOps", 2),
    "C14": ("Oracle Certified Professional Java SE", "Oracle", "Java", None),
    "C15": ("Machine Learning Specialization", "DeepLearning.AI", "Machine Learning", None),
    "C16": ("ITIL 4 Foundation", "PeopleCert", "Soporte técnico", None),
    "C17": ("Google UX Design Certificate", "Google", "Diseño UX/UI", None),
}

# id -> (nombre, habilidades que desarrolla, nivel, prerrequisitos, horas, descripción)
CURSOS = {
    "K01": ("SQL desde Cero", ["Bases de datos"], "básico", [], 20, "Consultas SELECT, filtros, uniones y funciones de agregación con SQL"),
    "K02": ("Curso SQL Avanzado", ["Bases de datos"], "avanzado", ["Bases de datos"], 30, "Optimización de consultas, índices, transacciones y procedimientos almacenados en SQL"),
    "K03": ("Administración de PostgreSQL", ["Bases de datos"], "avanzado", ["Bases de datos"], 40, "Respaldo, replicación, ajuste de rendimiento y seguridad en PostgreSQL"),
    "K04": ("Python Básico", ["Python"], "básico", [], 25, "Sintaxis, estructuras de datos, funciones y archivos en Python"),
    "K05": ("Python Intermedio", ["Python"], "intermedio", ["Python"], 30, "Programación orientada a objetos, módulos, pruebas y entornos virtuales en Python"),
    "K06": ("Pandas para análisis de datos", ["Análisis de datos", "Python"], "intermedio", ["Python"], 25, "Limpieza, transformación y análisis de tablas con pandas y Python"),
    "K07": ("Introducción a Machine Learning", ["Machine Learning"], "intermedio", ["Python"], 35, "Regresión, clasificación, validación y métricas con scikit-learn"),
    "K08": ("Proyecto práctico de análisis de datos", ["Análisis de datos"], "avanzado", ["Python", "Bases de datos"], 30, "Proyecto integrador de análisis de datos con SQL, pandas y visualización"),
    "K09": ("Visualización con Power BI", ["Análisis de datos"], "básico", [], 20, "Tableros, modelos de datos y medidas DAX en Power BI"),
    "K10": ("Docker y contenedores", ["DevOps"], "básico", [], 20, "Imágenes, contenedores, redes y volúmenes con Docker"),
    "K11": ("CI/CD con Jenkins", ["DevOps"], "intermedio", ["DevOps"], 25, "Pipelines de integración y despliegue continuo con Jenkins"),
    "K12": ("Fundamentos de AWS", ["Computación en la nube"], "básico", [], 25, "Servicios de cómputo, almacenamiento y redes en AWS"),
    "K13": ("Fundamentos de Azure", ["Computación en la nube"], "básico", [], 25, "Servicios de cómputo, datos e identidad en Azure"),
    "K14": ("Seguridad informática básica", ["Ciberseguridad"], "básico", [], 20, "Amenazas, controles y buenas prácticas de seguridad"),
    "K15": ("Preparación para CCNA", ["Redes"], "intermedio", [], 40, "Direccionamiento IP, enrutamiento y conmutación"),
    "K16": ("Scrum Master", ["Metodologías ágiles"], "intermedio", [], 16, "Roles, eventos y artefactos de Scrum"),
    "K17": ("Gestión de proyectos con Jira", ["Gestión de proyectos"], "básico", [], 12, "Tableros, backlogs y reportes en Jira"),
    "K18": ("Pruebas automatizadas con Selenium", ["Pruebas de software"], "intermedio", [], 25, "Automatización de pruebas web con Selenium"),
    "K19": ("Diseño de interfaces con Figma", ["Diseño UX/UI"], "básico", [], 18, "Prototipos y sistemas de diseño en Figma"),
    "K20": ("React desde cero", ["JavaScript"], "básico", [], 30, "Componentes, estado y consumo de APIs con React"),
    "K21": ("Node.js para backend", ["JavaScript"], "intermedio", ["JavaScript"], 28, "APIs REST y acceso a datos con Node.js"),
    "K22": ("Java con Spring Boot", ["Java"], "intermedio", [], 35, "Servicios REST y persistencia con Spring Boot"),
    "K23": ("ITIL 4 Foundation", ["Soporte técnico"], "básico", [], 16, "Gestión de servicios y de incidentes con ITIL 4"),
    "K24": ("Estadística aplicada a datos", ["Análisis de datos", "Machine Learning"], "intermedio", [], 24, "Estadística descriptiva e inferencial para análisis y modelos"),
}

DEPARTAMENTOS = [
    "Desarrollo de Software", "Datos y Analítica", "Infraestructura y Soporte",
    "Calidad y Seguridad", "Consultoría y Gestión de Proyectos",
]

CATEGORIAS_DOC = [
    "CV o perfil profesional", "Certificado o constancia", "Informe de cierre de proyecto",
    "Ticket de incidente", "Manual o política interna", "Minuta de reunión",
]

NIVELES = {1: "Básico", 2: "Elemental", 3: "Intermedio", 4: "Avanzado", 5: "Experto"}

# Reglas de coherencia (R1–R6 del avance) en lenguaje natural, para mostrarlas en la app
REGLAS = {
    "R1": "Una habilidad es «comprobada» solo si tiene al menos una evidencia vigente (certificación, proyecto, incidente o capacitación).",
    "R2": "Una certificación vencida no cuenta como evidencia ni suma al ranking.",
    "R3": "Todo empleado pertenece a un solo departamento; todo proyecto tiene cliente.",
    "R4": "Una participación en proyecto requiere rol y periodo válidos (inicio ≤ fin).",
    "R5": "Quien participó en un proyecto que usa una tecnología tiene experiencia probable en la habilidad asociada (pendiente de confirmación).",
    "R6": "Los sinónimos se asocian a un único nodo canónico de habilidad o tecnología.",
}
