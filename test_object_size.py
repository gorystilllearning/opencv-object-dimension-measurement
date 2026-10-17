import contextlib
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SCRIPTS = ("object_size.py", "object_size_mine.py")


class ObjectSizeTests(unittest.TestCase):
    def run_script(self, script, image_path, width):
        labels = []
        put_text = cv2.putText

        def draw_text(image, text, *args, **kwargs):
            labels.append(text)
            return put_text(image, text, *args, **kwargs)

        args = [script, "--image", str(image_path), "--width", str(width)]
        with patch("sys.argv", args), patch("cv2.imshow"), \
                patch("cv2.waitKey", return_value=0), patch("cv2.destroyAllWindows"), \
                patch("cv2.putText", side_effect=draw_text), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(ROOT / script), run_name="__main__")
        return labels

    def test_reference_and_double_size_object_are_labelled_in_cm(self):
        image = np.zeros((320, 480, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 50), (120, 150), (255, 255, 255), -1)
        cv2.rectangle(image, (200, 50), (400, 250), (255, 255, 255), -1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects.png"
            cv2.imwrite(str(path), image)
            for script in SCRIPTS:
                with self.subTest(script=script):
                    labels = self.run_script(script, path, 2.5)
                    self.assertEqual(labels, ["2.5 cm", "2.5 cm", "5.0 cm", "5.0 cm"])

    def test_width_must_be_positive_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(image, (40, 40), (140, 140), (255, 255, 255), -1)
            cv2.imwrite(str(path), image)
            for script in SCRIPTS:
                for width in (0, -1, "nan", "inf"):
                    with self.subTest(script=script, width=width):
                        with self.assertRaises(SystemExit) as error:
                            self.run_script(script, path, width)
                        self.assertEqual(error.exception.code, 2)

    def test_unreadable_image_has_cli_error(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                with self.assertRaises(SystemExit) as error:
                    self.run_script(script, ROOT / "nonexistent.png", 2.5)
                self.assertEqual(error.exception.code, 2)

    def test_image_without_objects_has_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            cv2.imwrite(str(path), np.zeros((200, 200, 3), dtype=np.uint8))
            for script in SCRIPTS:
                with self.subTest(script=script):
                    with self.assertRaises(SystemExit) as error:
                        self.run_script(script, path, 2.5)
                    self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
