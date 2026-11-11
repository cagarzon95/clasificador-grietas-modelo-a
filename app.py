import hmac
import io
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from PIL import Image

from azure_client import CustomVisionClient, PredictionError
from image_processing import ValidationError, read_safe_zip, validate_image

st.set_page_config(page_title="Clasificador de grietas", page_icon="🔎", layout="wide")

LABELS = {"crack": "Con grieta", "no_crack": "Sin grieta"}
RESULT_FIELDS = {
    "clase": "", "etiqueta_azure": "", "puntuacion_crack": "",
    "puntuacion_no_crack": "", "iteration_id": "", "prediction_id": "",
}


def setting(name: str, default=None):
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return os.getenv(name, default)


def authenticate() -> None:
    expected = str(setting("DEMO_PASSWORD", ""))
    if not expected:
        st.error("La aplicación no tiene configurada la contraseña de demostración.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    st.title("Acceso restringido")
    st.caption("Demostración académica protegida para limitar el consumo de Azure.")
    with st.form("login_form"):
        candidate = st.text_input("Contraseña de demostración", type="password")
        submitted = st.form_submit_button("Ingresar", type="primary")
    if submitted and hmac.compare_digest(candidate, expected):
        st.session_state.authenticated = True
        st.rerun()
    if submitted:
        st.error("Contraseña incorrecta.")
    st.stop()


def azure_client() -> CustomVisionClient:
    return CustomVisionClient(
        str(setting("AZURE_PREDICTION_ENDPOINT", "")), str(setting("AZURE_PREDICTION_KEY", "")),
        str(setting("AZURE_PROJECT_ID", "")), str(setting("AZURE_PUBLISHED_NAME", "modelo-a-g2")),
        str(setting("AZURE_ITERATION_ID", "")), float(setting("REQUEST_TIMEOUT_SECONDS", 30)),
    )


def remaining_quota() -> int:
    return max(0, int(setting("MAX_SESSION_PREDICTIONS", 25))
               - int(st.session_state.get("attempts_used", 0)))


def result_row(name: str, status: str, source: str, error: str = "", **values) -> dict:
    return {
        "archivo": name, "estado": status, **RESULT_FIELDS, **values, "origen": source,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "error": error,
    }


def classify(validated, source: str) -> dict:
    if remaining_quota() <= 0:
        raise PredictionError("La sesión alcanzó su límite de consultas.")
    prediction = azure_client().classify(validated.content)
    st.session_state.attempts_used = int(st.session_state.get("attempts_used", 0)) + prediction.attempts
    return result_row(
        validated.name, "clasificado", source, clase=LABELS[prediction.predicted_tag],
        etiqueta_azure=prediction.predicted_tag, puntuacion_crack=prediction.score_crack,
        puntuacion_no_crack=prediction.score_no_crack, iteration_id=prediction.iteration_id,
        prediction_id=prediction.prediction_id,
    )


def collect_batch(uploaded_files, maximum: int):
    images, errors = [], []
    for uploaded in uploaded_files:
        try:
            data = uploaded.getvalue()
            if uploaded.name.lower().endswith(".zip"):
                valid, invalid = read_safe_zip(uploaded.name, data, maximum)
                images.extend(valid)
                errors.extend(invalid)
            else:
                images.append(validate_image(uploaded.name, data))
        except ValidationError as exc:
            errors.append({"archivo": uploaded.name, "estado": "inválido", "error": str(exc)})
    return images, errors


def show_scores(row: dict) -> None:
    st.success(f"Resultado: **{row['clase']}**")
    left, right = st.columns(2)
    left.metric("Puntuación crack", f"{row['puntuacion_crack']:.4f}")
    right.metric("Puntuación no_crack", f"{row['puntuacion_no_crack']:.4f}")
    st.caption("Las puntuaciones son salidas del modelo y no probabilidades calibradas.")


authenticate()
st.title("Clasificador de grietas en imágenes")
st.caption("Modelo A · Azure Custom Vision · clasificación binaria")

with st.sidebar:
    st.subheader("Control de uso")
    st.write(f"Consultas disponibles en esta sesión: **{remaining_quota()}**")
    st.caption("Este límite es por sesión; el acceso restringido es el control principal.")
    if st.button("Cerrar sesión"):
        st.session_state.clear()
        st.rerun()

with st.expander("Alcance científico y limitaciones", expanded=True):
    st.markdown("""
    El **Modelo A** fue entrenado con HairWidthCracks, un conjunto de imágenes de
    hormigón. La validación interna obtuvo 223/225 aciertos, pero la evaluación
    externa mostró pérdida de desempeño ante cambios de material, textura,
    iluminación, escala y adquisición. Las puntuaciones no están calibradas.

    Esta aplicación clasifica la imagen completa; **no localiza ni segmenta**
    grietas y no es un sistema certificado de inspección industrial.
    """)

individual_tab, batch_tab = st.tabs(["Imagen individual", "Lote pequeño"])

with individual_tab:
    uploaded = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"], key="individual_file")
    validated = None
    if uploaded:
        try:
            validated = validate_image(uploaded.name, uploaded.getvalue())
            st.image(Image.open(io.BytesIO(validated.content)), caption=uploaded.name, width=520)
            st.caption(f"{validated.width} × {validated.height} px · {validated.image_format}")
        except ValidationError as exc:
            st.error(str(exc))
    if st.button("Clasificar", type="primary", disabled=validated is None, key="classify_individual"):
        try:
            with st.spinner("Consultando Modelo A…"):
                st.session_state.individual_result = classify(validated, "individual")
        except PredictionError as exc:
            st.error(str(exc))
    if st.session_state.get("individual_result"):
        show_scores(st.session_state.individual_result)

with batch_tab:
    st.write("Máximo 20 imágenes por operación. Se aceptan archivos individuales o un ZIP seguro.")
    batch_files = st.file_uploader("Seleccione imágenes o un ZIP", type=["jpg", "jpeg", "png", "zip"],
                                   accept_multiple_files=True, key="batch_files")
    if st.button("Clasificar lote", type="primary", disabled=not batch_files, key="classify_batch"):
        maximum = min(20, int(setting("MAX_BATCH_IMAGES", 20)))
        if len(batch_files) > maximum:
            st.error(f"La selección contiene más de {maximum} archivos.")
        else:
            candidates, invalid = collect_batch(batch_files, maximum)
            if len(candidates) > maximum:
                st.error(f"El lote expandido supera {maximum} imágenes.")
            elif len(candidates) > remaining_quota():
                st.error("El lote supera las consultas disponibles en esta sesión.")
            else:
                results, progress = [], st.progress(0, text="Procesando lote…")
                for index, candidate in enumerate(candidates, 1):
                    try:
                        results.append(classify(candidate, "lote"))
                    except PredictionError as exc:
                        results.append(result_row(candidate.name, "error", "lote", str(exc)))
                    progress.progress(index / max(len(candidates), 1), text=f"{index}/{len(candidates)}")
                progress.empty()
                results.extend(result_row(item["archivo"], item["estado"], "lote", item["error"])
                               for item in invalid)
                st.session_state.batch_results = results

    if st.session_state.get("batch_results"):
        frame = pd.DataFrame(st.session_state.batch_results)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        classified = int((frame["estado"] == "clasificado").sum())
        st.write(f"Clasificadas: **{classified}** · Fallos: **{len(frame) - classified}**")
        st.download_button("Descargar CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                           "resultados_modelo_a.csv", "text/csv")
