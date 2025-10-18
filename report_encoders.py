#!/usr/bin/env python3
"""
Genera un informe legible de los encoders usados por el pipeline (LabelEncoder/OneHotEncoder).

Salida (por defecto en app/models_cache/encoders_info/):
  - encoders_summary.json      → resumen machine-friendly
  - encoders_report.md         → informe humano con instrucciones
  - encoder_<col>.csv          → mapeos por columna (label ↔ code o categorías)

Uso:
  python report_encoders.py [--model app/models_cache/full_pipeline.pkl]
                            [--out-dir app/models_cache/encoders_info]
                            [--max-items 999]

Notas:
- Detecta encoders scaneando *app/models_cache/* por archivos *_encoder.pkl.
- Si el pipeline es RAW (InferencePreprocessor), tratará de listar las columnas crudas esperadas y
  (opcionalmente) las features generadas.
- Si encuentra feature_names.pkl, también las expone como "encoded_features" (modo ENCODED/fallback).
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# ----------------------------- FS helpers -----------------------------

def find_model_dir(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path.parent
    # fallback al default
    return Path("app/models_cache")


def scan_encoders(model_dir: Path) -> Dict[str, Any]:
    encoders: Dict[str, Any] = {}
    for p in sorted(model_dir.glob("*_encoder.pkl")):
        col = p.name.replace("_encoder.pkl", "")
        try:
            with p.open("rb") as f:
                enc = pickle.load(f)
            encoders[col] = enc
        except Exception as e:
            print(f"⚠️  No pude cargar {p}: {e}")
    return encoders


def try_load_feature_names(model_dir: Path) -> List[str]:
    p = model_dir / "feature_names.pkl"
    if p.exists():
        try:
            with p.open("rb") as f:
                names = pickle.load(f)
            if isinstance(names, (list, tuple)):
                return list(names)
        except Exception:
            pass
    return []


def try_load_pipeline(model_path: Path):
    try:
        with model_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def detect_mode_and_raw_inputs(pipe, model_dir: Path) -> Dict[str, Any]:
    schema = {"mode": "unknown"}
    if pipe is not None and hasattr(pipe, "named_steps"):
        pre = pipe.named_steps.get("pre")
        if pre is not None:
            pre_cls = type(pre).__name__
            if pre_cls == "FeatureAligner":
                schema["mode"] = "encoded"
            else:
                schema["mode"] = "raw"
                # intentar listar columnas crudas desde el inner preparer
                try:
                    if hasattr(pre, "_get_inner"):
                        inner = pre._get_inner()  # EnhancedDataPreparer instanciado dentro
                        if hasattr(inner, "list_expected_raw_inputs"):
                            schema["raw_input_columns"] = list(inner.list_expected_raw_inputs())
                except Exception:
                    pass

    # fallback: si hay feature_names.pkl, exponer encoded_features
    feats = try_load_feature_names(model_dir)
    if feats:
        schema.setdefault("encoded_features", feats)
    return schema


# ----------------------------- Encoder extraction -----------------------------

def encoder_info(enc: Any) -> Tuple[str, Dict[str, Any]]:
    """Devuelve (tipo, datos) para un encoder soportado."""
    if hasattr(enc, "classes_"):
        # sklearn LabelEncoder u objeto con .classes_
        classes = list(enc.classes_)
        return "LabelEncoder", {"classes": classes}
    if isinstance(enc, dict) and "classes_" in enc:
        classes = list(enc["classes_"])
        return "LabelEncoder", {"classes": classes}

    if hasattr(enc, "categories_"):
        cats = enc.categories_
        # normalizar a lista de listas (soportar 1 col o multi-col)
        if isinstance(cats, (list, tuple)) and len(cats) == 1:
            cats = [list(cats[0])]
        else:
            cats = [list(c) for c in cats]
        return "OneHotEncoder", {"categories": cats}
    if isinstance(enc, dict) and "categories_" in enc:
        cats = enc["categories_"]
        if isinstance(cats, (list, tuple)) and len(cats) == 1:
            cats = [list(cats[0])]
        else:
            cats = [list(c) for c in cats]
        return "OneHotEncoder", {"categories": cats}

    return type(enc).__name__, {}


def write_per_encoder_csv(out_dir: Path, col: str, etype: str, data: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    if etype == "LabelEncoder":
        rows = [{"label": lbl, "code": i} for i, lbl in enumerate(data.get("classes", []))]
        pd.DataFrame(rows).to_csv(out_dir / f"encoder_{col}.csv", index=False)
    elif etype == "OneHotEncoder":
        # Si es 1-col, guardamos una sola lista de categorías
        cats_all = data.get("categories", [])
        # concatenamos indicando el índice de la columna de entrada
        rows = []
        for j, cats in enumerate(cats_all):
            for c in cats:
                rows.append({"input_col_index": j, "category": c})
        pd.DataFrame(rows).to_csv(out_dir / f"encoder_{col}.csv", index=False)
    else:
        # otro tipo → guardamos JSON minimal
        pd.DataFrame([{"info": "unsupported encoder type"}]).to_csv(out_dir / f"encoder_{col}.csv", index=False)


# ----------------------------- Reporte principal -----------------------------

def build_report(model_path: Path, out_dir: Path, max_items: int = 999) -> Path:
    model_dir = find_model_dir(model_path)
    pipe = try_load_pipeline(model_path)

    schema = detect_mode_and_raw_inputs(pipe, model_dir)

    encoders = scan_encoders(model_dir)
    summary: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "mode": schema.get("mode", "unknown"),
        "raw_input_columns": schema.get("raw_input_columns", []),
        "encoded_features": schema.get("encoded_features", []),
        "encoders": {},
    }

    # CSVs por encoder + construir summary
    per_encoder_dir = out_dir
    per_encoder_dir.mkdir(parents=True, exist_ok=True)

    for col, enc in encoders.items():
        etype, data = encoder_info(enc)
        summary["encoders"][col] = {"type": etype, **data}
        write_per_encoder_csv(per_encoder_dir, col, etype, data)

    # Markdown humano
    md_path = out_dir / "encoders_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Informe de Encoders\n\n")
        f.write(f"Directorio del modelo: `{model_dir}`\n\n")
        f.write(f"**Modo detectado:** {summary['mode'].upper()}\n\n")
        if summary["mode"] == "raw":
            if summary["raw_input_columns"]:
                f.write("**Columnas RAW mínimas esperadas:**\n\n")
                for c in summary["raw_input_columns"]:
                    f.write(f"- `{c}`\n")
                f.write("\n")
            f.write(
                "> Los valores crudos se transforman internamente usando encoders persistidos.\n"
                "> Cualquier valor **no visto** durante el entrenamiento se mapea a `code=0` (fallback).\n\n"
            )
        else:
            if summary["encoded_features"]:
                f.write("**Features ENCODED esperadas (orden):**\n\n")
                for c in summary["encoded_features"][:max_items]:
                    f.write(f"- `{c}`\n")
                if len(summary["encoded_features"]) > max_items:
                    f.write(f"- ... (+{len(summary['encoded_features']) - max_items} más)\n")
                f.write("\n")

        if encoders:
            f.write("## Mapeos por columna\n\n")
            for col, info in summary["encoders"].items():
                etype = info.get("type", "?")
                f.write(f"### `{col}` — {etype}\n\n")
                csv_name = f"encoder_{col}.csv"
                f.write(f"Archivo: `{csv_name}`\n\n")
                if etype == "LabelEncoder":
                    classes = info.get("classes", [])
                    show = classes[:max_items]
                    f.write(f"Valores conocidos ({len(classes)}):\n\n")
                    for i, lbl in enumerate(show):
                        f.write(f"- code={i} → label=`{lbl}`\n")
                    if len(classes) > max_items:
                        f.write(f"- ... (+{len(classes) - max_items} más)\n")
                    f.write("\n")
                    f.write(
                        "> **Importante:** si aparece un valor que NO esté en esta lista, se asignará `code=0`.\n\n"
                    )
                elif etype == "OneHotEncoder":
                    cats_all = info.get("categories", [])
                    for j, cats in enumerate(cats_all):
                        f.write(f"Entrada {j}: categorías ({len(cats)}): ")
                        f.write(", ".join(f"`{c}`" for c in cats[:max_items]))
                        if len(cats) > max_items:
                            f.write(f" … (+{len(cats)-max_items})")
                        f.write("\n\n")
                else:
                    f.write("(Tipo de encoder no soportado para vista previa).\n\n")
        else:
            f.write("No se encontraron archivos *_encoder.pkl en el directorio del modelo.\n")

        # Guía rápida para usuarios RAW
        f.write("\n---\n\n")
        f.write("### Guía rápida (modo RAW)\n\n")
        f.write("1. Cree un CSV con las columnas listadas en **Columnas RAW mínimas esperadas**.\n")
        f.write("2. Para `driver`, `team`, `race_name`, `circuit_type`, use alguno de los labels listados en los CSV `encoder_*.csv`.\n")
        f.write("3. Si usa un valor no listado, internamente se mapeará a `code=0`.\n")

    # JSON resumen
    json_path = out_dir / "encoders_summary.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=2)

    print(f"📄 Reporte MD: {md_path}")
    print(f"🧾 Resumen JSON: {json_path}")
    print(f"📑 CSVs por encoder en: {per_encoder_dir}")
    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="app/models_cache/full_pipeline.pkl", help="Ruta al pipeline pickled")
    ap.add_argument("--out-dir", default="app/models_cache/encoders_info", help="Directorio de salida del informe")
    ap.add_argument("--max-items", type=int, default=999, help="Máximo de items a previsualizar en MD")
    args = ap.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    build_report(model_path, out_dir, max_items=args.max_items)

if __name__ == "__main__":
    main()
