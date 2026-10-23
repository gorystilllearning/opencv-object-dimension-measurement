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

import object_size

ROOT = Path(__file__).resolve().parent
SCRIPT = "object_size.py"


def make_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 100), (150, 200), (255, 255, 255), -1)
    cv2.rectangle(frame, (250, 100), (450, 200), (255, 255, 255), -1)
    cv2.circle(frame, (550, 150), 30, (255, 255, 255), -1)
    return frame


class ObjectSizeTests(unittest.TestCase):
    def run_script(self, cli_args, frames, keys=None):
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
        self.camera = mock_cap
        if keys is None:
            keys = [ord("q")]

        argv = [SCRIPT] + cli_args
        with patch("sys.argv", argv), \
             patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", side_effect=keys), \
             patch("cv2.destroyAllWindows"), \
             patch("cv2.putText", side_effect=capture_text), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            try:
                runpy.run_path(str(ROOT / SCRIPT), run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
        return labels

    def test_dimensions_calculated_with_correct_panjang_and_lebar(self):
        all_labels = self.run_script(["--scale", "20.0"], [make_frame()])
        cm_labels = [l for l in all_labels if "cm" in l and "Kelompok" not in l and "Skala" not in l]
        self.assertEqual(len(cm_labels), 4)
        self.assertTrue(any("P: " in l for l in cm_labels))
        self.assertTrue(any("L: " in l for l in cm_labels))
        self.assertEqual([float(label.split()[1]) for label in cm_labels], [5.0, 5.0, 10.0, 5.0])

    def test_scale_must_be_positive(self):
        for scale in ("0", "-5", "nan", "inf"):
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
        mock_cap.release.assert_called_once()

    def test_image_without_objects_labels_only_kelompok(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        all_labels = self.run_script([], [blank])
        cm_labels = [l for l in all_labels if "cm" in l and "Kelompok" not in l and "Skala" not in l]
        self.assertEqual(cm_labels, [])

    def test_reference_length_must_be_positive_and_finite(self):
        for option in ("--ref", "--width"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(option=option, value=value):
                    with self.assertRaises(SystemExit) as error:
                        self.run_script([option, value], [])
                    self.assertEqual(error.exception.code, 2)

    def test_calibration_uses_long_side_of_largest_object(self):
        frame = make_frame()
        labels = self.run_script(["--scale", "10", "--ref", "10"], [frame, frame],
                                 keys=[ord("k"), ord("q")])
        dimensions = [float(label.split()[1]) for label in labels if label.startswith(("P: ", "L: "))]
        self.assertEqual(dimensions, [10.0, 10.0, 20.0, 10.0, 5.0, 5.0, 10.0, 5.0])

    def test_calibration_stays_fixed_after_reference_leaves_frame(self):
        frame = make_frame()
        without_reference = frame.copy()
        without_reference[:, 180:] = 0
        labels = self.run_script(["--scale", "10", "--ref", "10"], [frame, without_reference],
                                 keys=[ord("k"), ord("q")])
        dimensions = [float(label.split()[1]) for label in labels if label.startswith(("P: ", "L: "))]
        self.assertEqual(dimensions[-2:], [5.0, 5.0])

    def test_manual_scale_adjustments_change_measurements(self):
        frame = make_frame()
        labels = self.run_script(["--scale", "20"], [frame, frame, frame],
                                 keys=[ord("+"), ord("-"), ord("q")])
        lengths = [float(label.split()[1]) for label in labels if label.startswith("P: ")]
        self.assertEqual(lengths, [5.0, 10.0, 4.9, 9.8, 5.0, 10.0])

    def test_read_failure_returns_error_and_releases_camera(self):
        with self.assertRaises(SystemExit) as error:
            self.run_script([], [])
        self.assertEqual(error.exception.code, 1)
        self.camera.release.assert_called_once()

    def test_render_error_still_releases_camera(self):
        with patch("cv2.drawContours", side_effect=RuntimeError("render failed")):
            with self.assertRaises(RuntimeError):
                self.run_script([], [make_frame()])
        self.camera.release.assert_called_once()

    def test_negative_camera_index_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.run_script(["--camera", "-1"], [])
        self.assertEqual(error.exception.code, 2)


class MeasurementLogicTests(unittest.TestCase):
    def test_rotated_box_preserves_fractional_side_lengths(self):
        for angle in (0, 35, 90, 145):
            with self.subTest(angle=angle):
                box = cv2.boxPoints(((250, 200), (120.5, 60.25), angle))
                length, width = object_size.box_sides(box)
                self.assertAlmostEqual(length, 120.5, places=4)
                self.assertAlmostEqual(width, 60.25, places=4)

    def test_invalid_boxes_are_rejected(self):
        for box in (np.zeros((4, 2)), np.zeros((3, 2)), np.full((4, 2), np.nan)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                object_size.box_sides(box)

    def test_rectangle_angle_check_rejects_degenerate_and_skewed_points(self):
        self.assertTrue(object_size.is_rectangular(cv2.boxPoints(((100, 100), (80, 40), 35))))
        for points in (np.zeros((4, 2)), np.full((4, 2), np.inf),
                       np.array([[0, 0], [100, 0], [180, 40], [0, 40]])):
            with self.subTest(points=points):
                self.assertFalse(object_size.is_rectangular(points))

    def test_calibration_uses_largest_object_not_leftmost(self):
        boxes = [{"box": cv2.boxPoints(((50, 50), (50, 30), 0)), "area": 1500},
                 {"box": cv2.boxPoints(((250, 150), (200, 100), 35)), "area": 20000}]
        self.assertAlmostEqual(object_size.calibrate_scale(boxes, 5), 40, places=4)

    def test_calibration_requires_a_valid_reference_and_detected_box(self):
        box = {"box": cv2.boxPoints(((100, 100), (80, 40), 0)), "area": 3200}
        for length in (0, -1, np.nan, np.inf):
            with self.subTest(length=length), self.assertRaises(ValueError):
                object_size.calibrate_scale([box], length)
        with self.assertRaises(ValueError):
            object_size.calibrate_scale([], 5)

    def test_detection_handles_light_and_dark_backgrounds(self):
        for frame in (make_frame(), cv2.bitwise_not(make_frame())):
            with self.subTest(background=int(frame[0, 0, 0])):
                boxes = object_size.detect_boxes(frame)
                self.assertEqual(len(boxes), 2)
                self.assertEqual([round(object_size.box_sides(item["box"])[0]) for item in boxes],
                                 [100, 200])

    def test_detection_ignores_small_rectangle_large_circle_and_full_frame(self):
        small = np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(small, (50, 50), (70, 70), (255, 255, 255), -1)
        circle = np.zeros_like(small)
        cv2.circle(circle, (250, 200), 80, (255, 255, 255), -1)
        full = np.zeros_like(small)
        cv2.rectangle(full, (5, 5), (635, 475), (255, 255, 255), -1)
        for frame in (small, circle, full):
            with self.subTest(frame=frame.shape):
                self.assertEqual(len(object_size.detect_boxes(frame)), 0)

    def test_extent_setting_is_used(self):
        with patch.object(object_size, "MIN_EXTENT", 1.1):
            self.assertEqual(len(object_size.detect_boxes(make_frame())), 0)

    def test_drawing_does_not_change_original_frame(self):
        frame = make_frame()
        original = frame.copy()
        result = object_size.draw_frame(frame, object_size.detect_boxes(frame), 20, "Skala manual")
        np.testing.assert_array_equal(frame, original)
        self.assertFalse(np.array_equal(result, original))

    def test_width_option_keeps_precedence_over_ref(self):
        _, args = object_size.parse_args(["--ref", "8.5", "--width", "5"])
        self.assertEqual(args.reference_length, 5)

    def test_reference_length_alias_is_supported(self):
        _, args = object_size.parse_args(["--reference-length", "5"])
        self.assertEqual(args.reference_length, 5)


if __name__ == "__main__":
    unittest.main()
