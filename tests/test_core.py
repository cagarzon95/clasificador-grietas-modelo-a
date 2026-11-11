import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from azure_client import CustomVisionClient, PredictionError
from image_processing import MAX_IMAGE_BYTES, ValidationError, read_safe_zip, validate_image


def image_bytes(fmt="PNG", size=(256, 256)):
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=fmt)
    return output.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_valid_sizes(self):
        for size in ((256, 256), (224, 224)):
            with self.subTest(size=size):
                result = validate_image("sample.png", image_bytes(size=size))
                self.assertEqual((result.width, result.height), size)

    def test_rejections(self):
        cases = (("invalid.png", b"not an image", "imagen válida"),
                 ("large.png", b"x" * (MAX_IMAGE_BYTES + 1), "4 MiB"),
                 ("extreme.png", image_bytes(size=(3000, 100)), "25:1"))
        for name, content, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValidationError, message):
                validate_image(name, content)

    def test_zip_blocks_traversal_and_keeps_valid(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("valid.png", image_bytes())
            archive.writestr("../escape.png", image_bytes())
        images, errors = read_safe_zip("batch.zip", output.getvalue(), 20)
        self.assertEqual((len(images), len(errors)), (1, 1))
        self.assertIn("insegura", errors[0]["error"])


class AzureClientTests(unittest.TestCase):
    def setUp(self):
        self.client = CustomVisionClient("https://example.test", "x" * 64, "project",
                                         "publish", "iteration", 1)

    @staticmethod
    def response(status=200, iteration="iteration"):
        response = Mock(status_code=status, raw=None)
        response.json.return_value = {
            "id": "prediction", "iteration": iteration,
            "predictions": [{"tagName": "crack", "probability": 0.9},
                            {"tagName": "no_crack", "probability": 0.1}],
        }
        return response

    @patch("azure_client.requests.Session.post")
    def test_parses_real_shape_and_uses_nostore(self, post):
        post.return_value = self.response()
        result = self.client.classify(image_bytes())
        self.assertEqual((result.predicted_tag, result.attempts), ("crack", 1))
        self.assertIn("/image/nostore", post.call_args.args[0])
        self.assertEqual(self.client.session.get_adapter("https://").max_retries.total, 2)

    @patch("azure_client.requests.Session.post")
    def test_rejects_wrong_iteration(self, post):
        post.return_value = self.response(iteration="wrong")
        with self.assertRaises(PredictionError):
            self.client.classify(image_bytes())

    @patch("azure_client.requests.Session.post")
    def test_reports_authentication_error_without_body(self, post):
        post.return_value = self.response(status=401)
        with self.assertRaisesRegex(PredictionError, "autenticación"):
            self.client.classify(image_bytes())


if __name__ == "__main__":
    unittest.main()
