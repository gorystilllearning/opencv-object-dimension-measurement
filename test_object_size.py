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

ROOT    = Path(__file__).resolve().parent
SCRIPTS = ("object_size.py", "object_size_mine.py")


def make_frame():
    frame = np.zeros((320, 480, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 50),  (120, 150), (255, 255, 255), -1)
    cv2.rectangle(frame, (200, 50), (400, 250), (255, 255, 255), -1)
    cv2.circle(frame, (450, 50), 10, (255, 255, 255), -1)
    return frame


class ObjectSizeTests(unittest.TestCase):
    def run_script(self, script, width, frames):
        labels   = []
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

        argv = [script, "--width", str(width)]
        with patch("sys.argv", argv), \
             patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.destroyWindow"), \
             patch("cv2.putText", side_effect=capture_text), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(ROOT / script), run_name="__main__")
        return labels

    def test_reference_and_double_size_object_are_labelled_in_cm(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                all_labels = self.run_script(script, 2.5, [make_frame()])
                cm_labels  = [l for l in all_labels if "cm" in l and "Kelompok" not in l]
                self.assertEqual(cm_labels, ["2.5 cm", "2.5 cm", "5.0 cm", "5.0 cm"])

    def test_width_must_be_positive_and_finite(self):
        for script in SCRIPTS:
            for width in (0, -1, "nan", "inf"):
                with self.subTest(script=script, width=width):
                    with self.assertRaises(SystemExit) as ctx:
                        self.run_script(script, width, [])
                    self.assertEqual(ctx.exception.code, 2)

    def test_camera_not_opened_has_cli_error(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        for script in SCRIPTS:
            with self.subTest(script=script):
                with patch("cv2.VideoCapture", return_value=mock_cap), \
                     patch("sys.argv", [script, "--width", "2.5"]), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        runpy.run_path(str(ROOT / script), run_name="__main__")
                    self.assertEqual(ctx.exception.code, 2)

    def test_image_without_objects_labels_only_kelompok(self):
        blank = np.zeros((320, 480, 3), dtype=np.uint8)
        for script in SCRIPTS:
            with self.subTest(script=script):
                all_labels = self.run_script(script, 2.5, [blank])
                cm_labels  = [l for l in all_labels if "cm" in l and "Kelompok" not in l]
                self.assertEqual(cm_labels, [])


if __name__ == "__main__":
    unittest.main()