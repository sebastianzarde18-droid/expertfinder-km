# Guía de la prueba de usabilidad (≈ 3 minutos por persona)

Abre https://expertfinder-km.streamlit.app/ y deja que la persona lo use sola. No le expliques nada de la interfaz.
Cronometra cada tarea (segundos) desde que lee la instrucción hasta que dice que terminó. Si pasa de 3 minutos, anota
el tiempo y marca `no` en «logro».

Tareas (léelas tal cual):
1. «Encuentra a una persona con experiencia en Python y bases de datos.»  → logro = nombra a alguien del ranking.
2. «Dime por qué esa persona aparece en el ranking.»  → logro = señala la evidencia (certificación, proyecto o incidente).
3. «Pregunta al asistente quién resolvió el incidente INC-0457.»  → logro = lee «Luis Pérez».
4. «Abre el grafo de una persona y dime con qué habilidades está conectada.»  → logro = menciona al menos 2 habilidades.

Al terminar pídele, de 1 (muy mal) a 5 (muy bien): facilidad de uso, claridad de los resultados y facilidad para
moverse entre pestañas. Anota un comentario si dice algo útil.

Llena `registro_usabilidad.csv` (logro = si / no) y ejecuta:  python -m src.eval_usabilidad
