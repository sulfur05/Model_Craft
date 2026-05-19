from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import joblib
import pandas as pd
import streamlit as st


EXPORT_ROOT = Path("exports")
BUNDLE_FILENAME = "model_bundle.joblib"
METADATA_FILENAME = "metadata.json"

REQUIRED_BUNDLE_KEYS = {
    "model",
    "preprocessor",
    "feature_columns",
    "target_column",
    "task_type",
    "trained_model_name",
    "created_at",
    "version",
    "dataset_shape",
    "metrics",
    "library_versions",
}


def _safe_name(name: str) -> str:
    """Make a filesystem-safe model name."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "model"


def _timestamp_version() -> str:
    """Create a sortable version string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _library_versions() -> Dict[str, str]:
    """Record the main library versions used to create the bundle."""
    versions: Dict[str, str] = {}
    for name, module in {
        "streamlit": st,
        "pandas": pd,
        "joblib": joblib,
    }.items():
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def _ensure_export_dir(model_name: str, version: str, root: Union[str, Path] = EXPORT_ROOT) -> Path:
    """Create and return the export directory for a given model/version."""
    export_dir = Path(root) / _safe_name(model_name) / version
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def build_model_bundle(
    model_name: Optional[str] = None,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete export bundle from Streamlit session state.

    This is the single source of truth for export.
    It packages the trained estimator, preprocessing pipeline, schema, and metadata.
    """
    required_session_keys = ["trained_model", "preprocessor", "feature_columns"]
    missing = [key for key in required_session_keys if key not in st.session_state]
    if missing:
        raise ValueError(
            "Missing required session_state values: " + ", ".join(missing)
        )

    model = st.session_state["trained_model"]
    preprocessor = st.session_state["preprocessor"]
    feature_columns = st.session_state["feature_columns"]
    target_column = st.session_state.get("target_column")
    task_type = st.session_state.get("task_type", "classification")
    trained_model_name = model_name or st.session_state.get("trained_model_name", type(model).__name__)
    created_at = datetime.now().isoformat(timespec="seconds")
    bundle_version = version or _timestamp_version()

    metrics = st.session_state.get("trained_model_metrics", {})
    comparison_results = st.session_state.get("model_comparison_results")
    params = None
    if hasattr(model, "get_params"):
        try:
            params = model.get_params()
        except Exception:
            params = None

    dataset = st.session_state.get("dataset")
    dataset_shape = tuple(dataset.shape) if hasattr(dataset, "shape") else None

    bundle: Dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "task_type": task_type,
        "trained_model_name": trained_model_name,
        "created_at": created_at,
        "version": bundle_version,
        "dataset_shape": dataset_shape,
        "metrics": metrics,
        "comparison_results": comparison_results,
        "params": params,
        "library_versions": _library_versions(),
    }

    return bundle


def validate_model_bundle(bundle: Dict[str, Any]) -> None:
    """Validate the bundle structure before save/load/use."""
    if not isinstance(bundle, dict):
        raise TypeError("Model bundle must be a dictionary.")

    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle.keys()))
    if missing:
        raise ValueError("Bundle is missing required keys: " + ", ".join(missing))

    if not isinstance(bundle["feature_columns"], (list, tuple)):
        raise TypeError("bundle['feature_columns'] must be a list or tuple.")

    if not isinstance(bundle["trained_model_name"], str):
        raise TypeError("bundle['trained_model_name'] must be a string.")


def save_model_bundle(
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    root: Union[str, Path] = EXPORT_ROOT,
) -> Dict[str, Any]:
    """
    Save the current trained model + preprocessor bundle to disk.

    Returns a dictionary containing the saved paths and metadata.
    """
    bundle = build_model_bundle(model_name=model_name, version=version)
    validate_model_bundle(bundle)

    export_dir = _ensure_export_dir(bundle["trained_model_name"], bundle["version"], root=root)
    bundle_path = export_dir / BUNDLE_FILENAME
    metadata_path = export_dir / METADATA_FILENAME

    joblib.dump(bundle, bundle_path)

    metadata = {
        "trained_model_name": bundle["trained_model_name"],
        "version": bundle["version"],
        "created_at": bundle["created_at"],
        "task_type": bundle["task_type"],
        "target_column": bundle["target_column"],
        "dataset_shape": bundle["dataset_shape"],
        "feature_count": len(bundle["feature_columns"]),
        "metrics": bundle["metrics"],
        "params": bundle["params"],
        "bundle_path": str(bundle_path),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "bundle": bundle,
        "bundle_path": bundle_path,
        "metadata_path": metadata_path,
        "export_dir": export_dir,
    }


def load_model_bundle(
    source: Union[str, Path, Any],
    persist_to_session: bool = True,
) -> Dict[str, Any]:
    """
    Load a previously saved model bundle.

    `source` can be a file path or a file-like object, including Streamlit's UploadedFile.
    """
    bundle = joblib.load(source)
    validate_model_bundle(bundle)

    if persist_to_session:
        restore_bundle_to_session(bundle)

    return bundle


def restore_bundle_to_session(bundle: Dict[str, Any]) -> None:
    """Put a loaded bundle back into Streamlit session_state so the app can use it immediately."""
    st.session_state["trained_model"] = bundle["model"]
    st.session_state["preprocessor"] = bundle["preprocessor"]
    st.session_state["feature_columns"] = list(bundle["feature_columns"])
    st.session_state["target_column"] = bundle.get("target_column")
    st.session_state["task_type"] = bundle.get("task_type", "classification")
    st.session_state["trained_model_name"] = bundle.get(
        "trained_model_name",
        type(bundle["model"]).__name__,
    )

    if bundle.get("metrics") is not None:
        st.session_state["trained_model_metrics"] = bundle["metrics"]

    if bundle.get("comparison_results") is not None:
        st.session_state["model_comparison_results"] = bundle["comparison_results"]


def list_saved_models(root: Union[str, Path] = EXPORT_ROOT) -> pd.DataFrame:
    """
    Return a registry table of saved model bundles found under the export directory.
    """
    root_path = Path(root)
    rows = []

    if not root_path.exists():
        return pd.DataFrame(
            columns=[
                "trained_model_name",
                "version",
                "created_at",
                "task_type",
                "target_column",
                "feature_count",
                "bundle_path",
            ]
        )

    for metadata_file in root_path.glob("*/*/metadata.json"):
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            rows.append(metadata)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "trained_model_name",
                "version",
                "created_at",
                "task_type",
                "target_column",
                "feature_count",
                "bundle_path",
            ]
        )

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["created_at", "trained_model_name", "version"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    return df


def _align_input_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    """
    Reorder and validate incoming batch prediction data to match training schema.
    """
    expected_columns = list(expected_columns)
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))

    aligned = df.copy()
    aligned = aligned[expected_columns]
    return aligned


def predict_with_bundle(
    bundle: Dict[str, Any],
    input_df: pd.DataFrame,
    include_proba: bool = True,
) -> pd.DataFrame:
    """
    Run batch prediction using a loaded bundle.

    Returns the original data plus prediction columns.
    """
    validate_model_bundle(bundle)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    expected_columns = bundle["feature_columns"]
    task_type = bundle.get("task_type", "classification")

    aligned_df = _align_input_columns(input_df, expected_columns)
    transformed = preprocessor.transform(aligned_df)

    output = input_df.copy()
    predictions = model.predict(transformed)
    output["prediction"] = predictions

    if task_type == "classification" and include_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(transformed)
        classes = list(getattr(model, "classes_", []))
        if len(classes) == proba.shape[1]:
            for idx, class_label in enumerate(classes):
                output[f"proba_{class_label}"] = proba[:, idx]

    return output