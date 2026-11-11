from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class PredictionError(RuntimeError):
    """Error seguro para interfaz: nunca incluye claves ni cuerpos de respuesta."""


@dataclass(frozen=True)
class PredictionResult:
    predicted_tag: str
    score_crack: float
    score_no_crack: float
    prediction_id: str
    iteration_id: str
    attempts: int


class CustomVisionClient:
    def __init__(self, endpoint: str, key: str, project_id: str, published_name: str,
                 expected_iteration_id: str, timeout_seconds: float = 30.0):
        if not all((endpoint, key, project_id, published_name, expected_iteration_id)):
            raise PredictionError("La configuración de Azure está incompleta.")
        self.endpoint, self.key, self.project_id = endpoint.rstrip("/"), key.strip(), project_id.strip()
        self.published_name, self.expected_iteration_id = published_name.strip(), expected_iteration_id.strip()
        self.timeout_seconds, self.session = timeout_seconds, requests.Session()
        retry = Retry(total=2, connect=2, read=2, status=2, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods={"POST"},
                      raise_on_status=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    @property
    def url(self) -> str:
        return (f"{self.endpoint}/customvision/v3.1/prediction/{self.project_id}/classify/"
                f"iterations/{self.published_name}/image/nostore")

    def classify(self, image_bytes: bytes) -> PredictionResult:
        try:
            response = self.session.post(
                self.url, headers={"Prediction-Key": self.key, "Content-Type": "application/octet-stream"},
                data=image_bytes, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise PredictionError(f"No fue posible conectar con Azure: {exc.__class__.__name__}.") from exc
        retries = getattr(getattr(response, "raw", None), "retries", None)
        attempts = len(retries.history) + 1 if retries else 1
        if response.status_code != 200:
            if response.status_code == 400:
                message = "Azure rechazó el archivo por formato, dimensiones o tamaño."
            elif response.status_code in {401, 403}:
                message = "Azure rechazó la autenticación o los permisos del recurso."
            elif response.status_code == 429:
                message = "Azure está temporalmente ocupado. Intente nuevamente en unos minutos."
            elif response.status_code in {500, 502, 503, 504}:
                message = "Azure presentó una falla temporal. Intente nuevamente."
            else:
                message = f"Azure respondió con estado HTTP {response.status_code}."
            raise PredictionError(message)
        try:
            payload = response.json()
            iteration_id = str(payload["iteration"])
            scores = {str(item["tagName"]): float(item["probability"])
                      for item in payload["predictions"]}
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionError("Azure devolvió una respuesta con formato inesperado.") from exc
        if iteration_id != self.expected_iteration_id:
            raise PredictionError("Azure respondió con una iteración distinta de Modelo A.")
        if set(scores) != {"crack", "no_crack"}:
            raise PredictionError("Azure devolvió etiquetas inesperadas.")
        return PredictionResult(max(scores, key=scores.get), scores["crack"], scores["no_crack"],
                                str(payload.get("id", "")), iteration_id, attempts)
