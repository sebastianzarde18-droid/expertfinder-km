"""Genera capturas de la interfaz y mide pasos por tarea (recorrido de usabilidad del equipo desarrollador)."""
import time, sys
from playwright.sync_api import sync_playwright
OUT = "figures/capturas/"
URL = "http://localhost:8599"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    pg.goto(URL); pg.wait_for_selector("text=Del conocimiento disperso", timeout=60000); time.sleep(2)
    pg.screenshot(path=OUT + "01_resumen.png")
    def tab(n): pg.get_by_role("tab", name=n).click(); time.sleep(1.5)
    tab("🔎 Especialistas")
    pg.get_by_role("textbox", name="Consulta").fill("Necesitamos una persona con experiencia en Python y bases de datos"); pg.keyboard.press("Enter"); time.sleep(3)
    pg.screenshot(path=OUT + "02_especialistas.png")
    pg.locator("summary", has_text="Emilio Cantú").first.click(); time.sleep(1.5)
    pg.screenshot(path=OUT + "03_especialistas_evidencia.png")
    tab("📄 Documentos")
    pg.locator("[data-testid=stSelectbox]").filter(has_text="Usar un ejemplo").locator("input").click(); time.sleep(.5)
    pg.get_by_role("option", name="Ejemplo: ticket de incidente").click(); time.sleep(1.5)
    pg.get_by_role("button", name="🔍 Analizar documento").click(); time.sleep(3)
    pg.screenshot(path=OUT + "04_documentos_clasificacion.png", full_page=True)
    tab("🎯 Recomendaciones"); time.sleep(1.5)
    pg.screenshot(path=OUT + "05_recomendacion_documentos.png")
    pg.get_by_role("tab", name="🎓 Capacitación personalizada").click(); time.sleep(2)
    pg.screenshot(path=OUT + "06_recomendacion_capacitacion.png", full_page=True)
    tab("💬 Asistente")
    for i, q in enumerate(["¿Quién resolvió el incidente INC-0457 y cómo?", "¿Cuál es el procedimiento para solicitar un especialista?", "¿Cuál es el procedimiento para solicitar un equipo de cómputo nuevo?"]):
        pg.get_by_test_id("stChatInputTextArea").fill(q); pg.keyboard.press("Enter"); time.sleep(3)
    pg.screenshot(path=OUT + "07_asistente.png", full_page=True)
    tab("🕸️ Grafo"); time.sleep(9); pg.mouse.wheel(0, 420); time.sleep(1)
    pg.screenshot(path=OUT + "08_grafo.png")
    tab("📊 Evaluación"); time.sleep(2)
    pg.screenshot(path=OUT + "09_evaluacion.png", full_page=True)
    b.close()
print("ok")
