# ExpertFinder · Directorio inteligente de conocimiento y talento (TalentTech Consulting)

Prototipo funcional de administración del conocimiento: grafo de conocimiento (ontología), clasificación automática de documentos,
búsqueda de especialistas con evidencia, recomendación (documentos y capacitación), asistente conversacional RAG y evaluación.
**Todos los datos son simulados.**

## Estructura
- `app.py` – interfaz Streamlit (7 pestañas)
- `src/` – ontología, datos, NLP, grafo, índice, clasificador, buscador, recomendador, chatbot, evaluación, diagramas
- `data/` – CSV y documentos simulados (se regeneran con `python -m src.data_gen`)
- `reports/`, `figures/` – métricas y gráficas generadas por `python -m src.evaluation` y `python -m src.diagrams`
- `notebooks/ExpertFinder_Colab.ipynb` – recorrido completo para Google Colab
- `tests/` – pruebas automáticas (`pytest`) y script de capturas

## Ejecución local
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_gen        # genera los datos
python -m src.evaluation      # métricas, tablas y figuras
python -m src.diagrams        # diagramas
streamlit run app.py
```

## Google Colab
Abre `notebooks/ExpertFinder_Colab.ipynb`, sube el zip del proyecto (o clona tu repositorio en la primera celda) y ejecuta todas las celdas.

## Publicar en Streamlit Community Cloud
1. Sube el repositorio a GitHub (público).
2. En https://share.streamlit.io elige *New app*, selecciona el repo, rama `main` y archivo principal `app.py`.
3. Copia el enlace público generado y colócalo en el documento y la presentación.

## Opcional
- LLM en el asistente: define `ANTHROPIC_API_KEY` (en Streamlit Cloud: *Secrets*). Sin clave usa modo extractivo.
- Embeddings neuronales: `pip install sentence-transformers` y `EF_EMBEDDINGS=st`.
