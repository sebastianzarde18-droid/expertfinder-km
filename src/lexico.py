"""Léxico de paráfrasis del dominio (parte de la ontología): grupos de palabras que se usan como equivalentes.

Se emplea para (1) ampliar la consulta antes de buscar y (2) reconocer que un término de la pregunta está cubierto
por el texto recuperado aunque use otra palabra (p. ej. «caducó» ≡ «vencida»). Es un recurso léxico acotado, hecho a mano:
no sustituye a un modelo de embeddings preentrenado, que es la mejora prevista para cubrir paráfrasis no listadas."""
from __future__ import annotations
from .nlp import normalizar

GRUPOS = [
    "caducar caducado caduco caduca vencer vencida vencido vence vencimiento expirar expira caducidad",
    "ley norma reglamento regula regulan protege proteger legislacion",
    "informacion datos",
    "empleado empleados colaborador colaboradores trabajador trabajadores",
    "falla fallas fallo incidente incidentes error errores ticket",
    "grave alta critica urgente",
    "reporte informe",
    "lapso plazo periodo tardar tarda",
    "pedir pido solicitar solicitud solicita pedido",
    "pagados pagado financiadas financiados financia financiar pagar",
    "curso cursos capacitacion formacion",
    "dominan domina dominar experiencia sabe conoce",
    "origen causa motivo",
    "ratificar confirmar confirma validar",
    "posicionarse ranking posicion puntos suma sirve cuenta",
    "avisar aviso avisa escalar escalamiento escala notificar",
    "buscar busco encontrar localizar",
    "lider jefe responsable",
]
_GRUPO_DE: dict[str, set[str]] = {}
for g in GRUPOS:
    palabras = set(normalizar(g).split())
    for w in palabras:
        _GRUPO_DE.setdefault(w, set()).update(palabras)


def _stem(w: str) -> str:
    for suf in ("es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def equivalentes(palabra: str) -> set[str]:
    """Palabras equivalentes a la dada (incluye ella misma)."""
    g = _GRUPO_DE.get(palabra) or _GRUPO_DE.get(_stem(palabra)) or set()
    return set(g) | {palabra}


def ampliar(consulta: str) -> list[str]:
    extra = []
    for w in normalizar(consulta).split():
        extra += sorted(equivalentes(w) - {w})
    return list(dict.fromkeys(extra))
