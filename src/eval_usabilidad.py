"""Resume usabilidad/registro_usabilidad.csv (datos reales de la prueba con personas) y genera la tabla de la sección 9.7."""
import pandas as pd
from pathlib import Path
RUTA = Path(__file__).resolve().parent.parent / "usabilidad"

def main():
    d = pd.read_csv(RUTA / "registro_usabilidad.csv")
    d = d.dropna(subset=["t1_seg"]).copy()
    if d.empty:
        print("Aún no hay datos registrados."); return
    out = {}
    for i in range(1, 5):
        ok = d[f"t{i}_logro"].str.lower().isin(["si", "sí"])
        out[f"Tarea {i}"] = dict(completada=f"{ok.sum()} de {len(d)}", tiempo_medio_s=round(d[f"t{i}_seg"].mean(), 1))
    res = pd.DataFrame(out).T
    prom = d[["facilidad_1a5", "claridad_1a5", "navegacion_1a5"]].mean().round(2)
    res.to_csv(RUTA / "resumen_usabilidad.csv")
    print(res.to_string()); print("\nPromedios (1-5):"); print(prom.to_string())
    print("\nTabla por participante (para el documento):")
    cols = ["participante"] + [f"t{i}_seg" for i in range(1, 5)] + ["facilidad_1a5", "claridad_1a5", "navegacion_1a5"]
    print(d[cols].to_markdown(index=False) if hasattr(d, "to_markdown") else d[cols].to_string(index=False))

if __name__ == "__main__":
    main()
