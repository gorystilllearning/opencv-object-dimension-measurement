# ==========================================================
# UTS Machine Vision - Pengukuran Dimensi Objek
# Kelompok 8
#   1. Ahmad Bagus Idkholus Surur   (23050874166)
#   2. Khairie Arrantisy            (23050874182)
#   3. Muhammad Ilham Gymnastiar    (23050874169)
#   4. Wahyu Rizki Agung Kurniawan  (23050874189)
# ==========================================================

import contextlib
import io
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SCRIPT = "object_size.py"


def make_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 100), (150, 200), (255, 255, 255), -1)
    cv2.rectangle(frame, (250, 100), (450, 200), (255, 255, 255), -1)
    cv2.circle(frame, (550, 150), 30, (255, 255, 255), -1)
    return frame


class ObjectSizeTests(unittest.TestCase):
    def run_script(self, cli_args, frames):
        labels = []
        orig_put = cv2.putText
        frame_iter = iter(frames)

        def fake_read():
            try:
                return True, next(frame_iter)
            except StopIteration:
                return False, None

        def capture_text(img, text, *a, **kw):
            labels.append(text)
            return orig_put(img, text, *a, **kw)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = fake_read

        argv = [SCRIPT] + cli_args
        with patch("sys.argv", argv), \
             patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.putText", side_effect=capture_text), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(ROOT / SCRIPT), run_name="__main__")
        return labels

    def test_dimensions_calculated_with_correct_panjang_and_lebar(self):
        all_labels = self.run_script(["--scale", "20.0"], [make_frame()])
        cm_labels = [l for l in all_labels if "cm" in l and "Kelompok" not in l and "Skala" not in l]
        self.assertEqual(len(cm_labels), 4)
        self.assertTrue(any("P: " in l for l in cm_labels))
        self.assertTrue(any("L: " in l for l in cm_labels))

    def test_scale_must_be_positive(self):
        for scale in ("0", "-5"):
            with self.subTest(scale=scale):
                with self.assertRaises(SystemExit) as ctx:
                    self.run_script(["--scale", scale], [])
                self.assertEqual(ctx.exception.code, 2)

    def test_camera_not_opened_has_cli_error(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("sys.argv", [SCRIPT]), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                runpy.run_path(str(ROOT / SCRIPT), run_name="__main__")
            self.assertEqual(ctx.exception.code, 2)

    def test_image_without_objects_labels_only_kelompok(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        all_labels = self.run_script([], [blank])
        cm_labels = [l for l in all_labels if "cm" in l and "Kelompok" not in l and "Skala" not in l]
        self.assertEqual(cm_labels, [])


if __name__ == "__main__":
    unittest.main()