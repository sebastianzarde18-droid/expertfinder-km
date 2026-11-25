import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.sistema import Sistema

S = Sistema()

def test_grafo_tiene_al_menos_30_nodos():
    assert S.kb.resumen()['nodos'] >= 30

def test_busqueda_devuelve_resultados_con_explicacion():
    r = S.buscador.buscar('experiencia en Python y bases de datos', k=3)
    assert r['resultados'] and r['resultados'][0]['explicacion']

def test_chatbot_rechaza_informacion_inexistente():
    r = S.asistente.responder('¿Cuál es el procedimiento para solicitar un equipo de cómputo nuevo?')
    assert r['suficiente'] is False

def test_chatbot_responde_incidente():
    r = S.asistente.responder('¿Quién resolvió el incidente INC-0457?')
    assert r['suficiente'] and r['fuentes']
