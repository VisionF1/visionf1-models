from __future__ import annotations
import os, pickle, unicodedata
from typing import Any
from .manifest import InferenceManifest

class EncoderResolver:
    def __init__(self, manifest: InferenceManifest):
        self.manifest = manifest
        self._cache: dict[str, Any] = {}

    def _log(self, msg: str):
        self.manifest.log(msg)

    @staticmethod
    def strip_accents(text: str) -> str:
        if text is None:
            return text
        text = unicodedata.normalize('NFD', text)
        return ''.join([c for c in text if unicodedata.category(c) != 'Mn'])

    @staticmethod
    def norm_up(x: str) -> str:
        if x is None:
            return x
        return EncoderResolver.strip_accents(str(x)).upper().strip()

    @staticmethod
    def norm_lo(x: str) -> str:
        if x is None:
            return x
        return EncoderResolver.strip_accents(str(x)).lower().strip()

    # --- carga/resolve ---
    def resolve(self, key: str):
        if key in self._cache:
            return self._cache[key]
        spec = self.manifest.get_encoder_spec(key)
        enc_obj = None
        if isinstance(spec, (dict, list, tuple)):
            enc_obj = spec
        elif isinstance(spec, str) and spec.endswith(".pkl"):
            path = spec
            if not os.path.exists(path):
                cand = os.path.join(self.manifest.manifest_dir(), os.path.basename(spec))
                if os.path.exists(cand):
                    path = cand
            try:
                with open(path, "rb") as f:
                    enc_obj = pickle.load(f)
                self._log(f"📦 Encoder '{key}' cargado (tipo={type(enc_obj).__name__})")
            except Exception as e:
                self._log(f"❌ No pude cargar encoder {key} desde {spec}: {e}")
                enc_obj = None
        else:
            enc_obj = spec
        self._cache[key] = enc_obj
        return enc_obj

    # --- encode/decode helpers ---
    def encode_value(self, key: str, raw_value):
        enc = self.resolve(key)
        if enc is None:
            return None
        # dict/list mapping
        if isinstance(enc, dict):
            return enc.get(raw_value)
        if isinstance(enc, (list, tuple)):
            try:
                return enc.index(raw_value)
            except ValueError:
                return None
        # LabelEncoder
        classes = getattr(enc, "classes_", None)
        if classes is not None:
            vals = [str(x) for x in list(classes)]
            try:
                return vals.index(str(raw_value))
            except ValueError:
                return None
        # OrdinalEncoder
        cats = getattr(enc, "categories_", None)
        if cats is not None:
            arr = cats[0] if len(cats) == 1 else cats[0]
            vals = [str(x) for x in list(arr)]
            try:
                return vals.index(str(raw_value))
            except ValueError:
                return None
        return None

    def decode_series(self, key: str, code_series):
        import pandas as pd
        enc = self.resolve(key)
        if enc is None:
            return pd.Series([None] * len(code_series), index=code_series.index)
        if isinstance(enc, dict):
            inv = {v: k for k, v in enc.items()}
            return code_series.map(inv.get)
        if isinstance(enc, (list, tuple)):
            arr = list(enc)
            def _map(i):
                try:
                    i = int(i); return arr[i] if 0 <= i < len(arr) else None
                except Exception:
                    return None
            return code_series.map(_map)
        classes = getattr(enc, "classes_", None)
        if classes is not None:
            vals = [str(x) for x in list(classes)]
            def _le(x):
                try:
                    i = int(x);  return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_le)
        cats = getattr(enc, "categories_", None)
        if cats is not None:
            arr = cats[0] if len(cats) == 1 else cats[0]
            vals = [str(x) for x in list(arr)]
            def _oe(x):
                try:
                    i = int(x);  return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_oe)
        return pd.Series([None] * len(code_series), index=code_series.index)

