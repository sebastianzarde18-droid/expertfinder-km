"""Procesamiento de lenguaje natural: normalización, extracción de entidades y propuestas.

La extracción es un NER basado en la ontología (equivalente a un EntityRuler de spaCy):
detecta menciones de habilidades, tecnologías, certificaciones, cursos, empleados, proyectos,
clientes e incidentes y las normaliza a su nombre canónico (regla R6).
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, asdict

from .ontology import HABILIDADES, TECNOLOGIAS, CERTIFICACIONES, CURSOS

STOPWORDS = set("""a al algo ante con como cual cuales cuando cuanto cuantos de del desde donde el ella ellos en entre es esta estan este esto
fue ha han hay la las le les lo los me mi mis muy no nos o para pero por que quien quienes se si sin sobre son su sus te tiene tienen
tu un una uno unas unos y ya necesitamos necesito persona personas experiencia experto expertos buscar busco hacer hago puedo pueden
puede debe deben hay cual cuales hoy tengo tenemos hago hacemos necesito debo debemos quiero quisiera especialista especialistas alguien perfil perfiles candidato candidatos sabe domina domine conoce trabaja""".split())


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, sin signos; conserva '.', '-', '/', '+', '#' dentro de términos técnicos."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    t = t.lower()
    t = re.sub(r"[^a-z0-9.\-/+# ]+", " ", t)
    t = re.sub(r"(?<![a-z0-9])[.\-/+#]+|[.\-/+#]+(?![a-z0-9])", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens_contenido(texto: str) -> list[str]:
    return [w for w in normalizar(texto).split() if w not in STOPWORDS and len(w) > 2]


@dataclass
class Entidad:
    tipo: str          # habilidad | tecnologia | certificacion | curso | empleado | proyecto | cliente | incidente
    canonico: str      # nombre canónico (R6)
    mencion: str       # texto tal cual aparece
    confianza: float   # 1.0 coincidencia exacta de nombre canónico; 0.9 sinónimo; 0.8 acrónimo


class Extractor:
    """Extractor de entidades guiado por la ontología y los catálogos de la organización."""

    def __init__(self, empleados=None, proyectos=None, clientes=None):
        # lista de (patrón normalizado, tipo, canónico, confianza)
        self.lexico: list[tuple[str, str, str, float]] = []
        for h, (_, sins, _) in HABILIDADES.items():
            self._add(h, "habilidad", h, 1.0)
            for s in sins:
                self._add(s, "habilidad", h, 0.9)
        for t, (_, _, sins) in TECNOLOGIAS.items():
            self._add(t, "tecnologia", t, 1.0)
            for s in sins:
                self._add(s, "tecnologia", t, 0.9)
        for cid, (nom, _, _, _) in CERTIFICACIONES.items():
            self._add(nom, "certificacion", cid, 1.0)
            for acr in re.findall(r"\(([A-Z0-9\-]{3,})\)", nom) + [nom.split(" ")[0] if nom.split(" ")[0].isupper() else ""]:
                if acr:
                    self._add(acr, "certificacion", cid, 0.8)
            # nombre sin la parte después de la raya
            self._add(re.split(r"\s[–-]\s", nom)[0], "certificacion", cid, 0.95)
        self._add("CCNA", "certificacion", "C06", 0.8)
        self._add("PMP", "certificacion", "C07", 0.8)
        for kid, (nom, *_rest) in CURSOS.items():
            self._add(nom, "curso", kid, 1.0)
        if empleados is not None:
            for r in empleados.itertuples():
                self._add(r.nombre, "empleado", r.id, 1.0)
        if proyectos is not None:
            for r in proyectos.itertuples():
                self._add(r.nombre, "proyecto", r.id, 1.0)
        for c in (clientes or []):
            self._add(c, "cliente", c, 1.0)
        # ordena por longitud para que gane la coincidencia más larga
        self.lexico.sort(key=lambda x: -len(x[0]))
        self._re_inc = re.compile(r"\bINC-\d{4}\b", re.I)

    def _add(self, patron, tipo, canonico, conf):
        p = normalizar(patron)
        if p:
            self.lexico.append((p, tipo, canonico, conf))

    def extraer(self, texto: str) -> list[Entidad]:
        norm = " " + normalizar(texto) + " "
        usados = []  # intervalos ya cubiertos
        encontrados: dict[tuple[str, str], Entidad] = {}
        for patron, tipo, canon, conf in self.lexico:
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(patron) + r"(?![a-z0-9])", norm):
                a, b = m.start(), m.end()
                if any(not (b <= x or a >= y) for x, y in usados):
                    continue
                usados.append((a, b))
                clave = (tipo, canon)
                if clave not in encontrados or conf > encontrados[clave].confianza:
                    encontrados[clave] = Entidad(tipo, canon, patron, conf)
        for m in self._re_inc.finditer(texto):
            encontrados[("incidente", m.group(0).upper())] = Entidad("incidente", m.group(0).upper(), m.group(0), 1.0)
        return list(encontrados.values())

    def extraer_dicts(self, texto: str) -> list[dict]:
        return [asdict(e) for e in self.extraer(texto)]


def habilidad_de(entidad: Entidad) -> str | None:
    """Habilidad canónica asociada a una entidad de tipo habilidad o tecnología (regla R5)."""
    if entidad.tipo == "habilidad":
        return entidad.canonico
    if entidad.tipo == "tecnologia":
        return TECNOLOGIAS[entidad.canonico][0]
    return None


RE_VIGENCIA = re.compile(r"vigencia hasta:?\s*(\d{2})/(\d{2})/(\d{4})", re.I)
RE_OBTENCION = re.compile(r"fecha de obtenci[oó]n:?\s*(\d{2})/(\d{2})/(\d{4})", re.I)


def fechas_certificado(texto: str) -> dict:
    out = {}
    m = RE_OBTENCION.search(texto)
    if m:
        out["obtencion"] = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = RE_VIGENCIA.search(texto)
    if m:
        out["vigencia"] = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return out
