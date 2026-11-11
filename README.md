# Clasificador de grietas — Modelo A

Aplicación Streamlit académica que consulta una iteración publicada de Azure Custom
Vision mediante el endpoint `image/nostore`. Clasifica imágenes individuales o lotes
pequeños y permite descargar resultados CSV.

## Ejecución local

1. Instale `requirements.txt`.
2. Copie `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y complete
   los secretos localmente. Este archivo está excluido de Git.
3. Ejecute `streamlit run app.py`.

## Seguridad

- No se almacenan imágenes ni se registran claves.
- La contraseña de demostración y las credenciales Azure viven exclusivamente en
  secretos de Streamlit o variables de entorno.
- Máximo 20 imágenes por lote y 25 intentos por sesión por defecto.
- El acceso debe mantenerse privado/restringido; el límite por sesión no sustituye
  una cuota global distribuida.

## Modelo

- Proyecto: `b15f3466-dc32-41d1-9182-a6564caeafe5`
- Publicación: `modelo-a-g2`
- Iteración: `9088ab6c-0c90-48c1-8cb6-b95920745d14`

El modelo fue entrenado con HairWidthCracks (hormigón). No es un sistema certificado
de inspección y sus puntuaciones no son probabilidades calibradas.

El límite de entrada es 4 MiB por imagen porque ese es el máximo vigente para
predicciones de Custom Vision v3.1. Las imágenes HairWidthCracks de 224×224 son
admitidas y el servicio las reescala. Referencia oficial:
<https://learn.microsoft.com/azure/ai-services/custom-vision-service/limits-and-quotas>.

