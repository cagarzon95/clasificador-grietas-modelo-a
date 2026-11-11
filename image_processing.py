import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS, ALLOWED_FORMATS = {".jpg", ".jpeg", ".png"}, {"JPEG", "PNG"}
MAX_IMAGE_BYTES, MAX_ZIP_BYTES = 4 * 1024**2, 25 * 1024**2
MAX_ZIP_UNCOMPRESSED_BYTES, MAX_COMPRESSION_RATIO = 80 * 1024**2, 100


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedImage:
    name: str
    content: bytes
    width: int
    height: int
    image_format: str


def validate_image(name: str, content: bytes) -> ValidatedImage:
    suffix = PurePosixPath(name.replace("\\", "/")).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValidationError("Extensión no permitida; utilice JPG, JPEG o PNG.")
    if not content:
        raise ValidationError("El archivo está vacío.")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValidationError("La imagen supera el límite efectivo de 4 MiB de Azure.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height, image_format = *image.size, str(image.format)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError("El archivo no contiene una imagen válida.") from exc
    if image_format not in ALLOWED_FORMATS:
        raise ValidationError("El contenido no es JPEG o PNG.")
    if max(width, height) > 10240:
        raise ValidationError("La imagen supera 10.240 píxeles por lado.")
    if max(width / height, height / width) > 25:
        raise ValidationError("La relación de aspecto supera 25:1.")
    return ValidatedImage(name, content, width, height, image_format)


def read_safe_zip(name: str, content: bytes, max_images: int):
    if not name.lower().endswith(".zip"):
        raise ValidationError("El archivo no es ZIP.")
    if len(content) > MAX_ZIP_BYTES:
        raise ValidationError("El ZIP comprimido supera 25 MiB.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValidationError("El ZIP está dañado o no es válido.") from exc
    images, errors = [], []
    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) > max_images:
            raise ValidationError(f"El ZIP contiene más de {max_images} archivos.")
        if sum(item.file_size for item in members) > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValidationError("El ZIP supera 80 MiB descomprimidos.")
        for item in members:
            normalized = item.filename.replace("\\", "/")
            path = PurePosixPath(normalized)
            try:
                if path.is_absolute() or ".." in path.parts or normalized.startswith("/"):
                    raise ValidationError("Ruta interna insegura.")
                if item.file_size / max(item.compress_size, 1) > MAX_COMPRESSION_RATIO:
                    raise ValidationError("Relación de compresión sospechosa.")
                images.append(validate_image(item.filename, archive.read(item)))
            except (ValidationError, RuntimeError, zipfile.BadZipFile) as exc:
                errors.append({"archivo": item.filename, "estado": "inválido", "error": str(exc)})
    return images, errors
