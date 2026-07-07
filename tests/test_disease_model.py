import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from utils import disease_model


class DiseaseModelFallbackTest(unittest.TestCase):
    def test_detect_crop_disease_returns_fallback_result_for_missing_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = Path(tmpdir) / "best_crop_disease_model.pth"
            with patch.object(disease_model, "MODEL_PATH", temp_model_path):
                img_bytes = BytesIO()
                Image.new("RGB", (64, 64), color=(20, 180, 40)).save(img_bytes, format="PNG")
                img_bytes.seek(0)

                result = disease_model.detect_crop_disease(img_bytes)

                self.assertTrue(result["available"])
                self.assertIn("healthy", result["prediction"].lower())
                self.assertIn("fallback", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
