# ==========================================================
# UTS Machine Vision - Pengukuran Dimensi Objek
# Kelompok 8
#   1. Ahmad Bagus Idkholus Surur   (23050874166)
#   2. Khairie Arrantisy            (23050874182)
#   3. Muhammad Ilham Gymnastiar    (23050874169)
#   4. Wahyu Rizki Agung Kurniawan  (23050874189)
# ==========================================================

import argparse

import cv2
import numpy as np


WINDOW = "Object Dimension Measurement"
BLUR_KERNEL = (7, 7)
MORPH_KERNEL = (9, 9)
MIN_CONTOUR_AREA = 1500
MAX_FRAME_AREA_RATIO = 0.85
APPROX_EPSILON_RATIO = 0.035
MIN_EXTENT = 0.70
MAX_ANGLE_COSINE = 0.35
SCALE_STEP = 0.5


def is_rectangular(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape not in ((4, 2), (4, 1, 2)) or not np.isfinite(pts).all():
        return False
    pts = pts.reshape(4, 2)
    for i in range(4):
        p0, p1, p2 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        v1, v2 = p0 - p1, p2 - p1
        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denominator == 0:
            return False
        cos_th = abs(np.dot(v1, v2) / denominator)
        if cos_th > MAX_ANGLE_COSINE:
            return False
    return True


def positive_number(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("nilai harus positif dan finite")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="UTS Machine Vision - Kelompok 8: pengukuran dimensi objek dengan OpenCV",
        epilog="1. Ahmad Bagus Idkholus Surur (23050874166); 2. Khairie Arrantisy (23050874182); "
               "3. Muhammad Ilham Gymnastiar (23050874169); 4. Wahyu Rizki Agung Kurniawan (23050874189)")
    parser.add_argument("-s", "--scale", type=positive_number, default=20.0,
                        help="skala awal/manual px/cm, bukan hasil kalibrasi (default: 20)")
    parser.add_argument("-r", "--ref", "--reference-length", dest="reference_length",
                        type=positive_number, default=8.5,
                        help="sisi terpanjang referensi dalam cm (default: 8.5)")
    parser.add_argument("-w", "--width", type=positive_number, default=None,
                        help="opsi lama untuk panjang referensi; menggantikan --ref")
    parser.add_argument("-c", "--camera", type=int, default=0, help="index kamera (default: 0)")
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera tidak boleh negatif")
    if args.width is not None:
        args.reference_length = args.width
    return parser, args


def box_sides(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4, 2) or not np.isfinite(box).all():
        raise ValueError("Koordinat kotak tidak valid")
    sides = sorted((float(np.linalg.norm(box[1] - box[0])),
                    float(np.linalg.norm(box[2] - box[1]))), reverse=True)
    if sides[1] <= 0:
        raise ValueError("Panjang sisi kotak harus positif")
    return sides


def detect_boxes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh.mean() > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > MAX_FRAME_AREA_RATIO * frame_area:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, APPROX_EPSILON_RATIO * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx) or not is_rectangular(approx):
            continue
        rect = cv2.minAreaRect(hull)
        bw, bh = rect[1]
        if bw == 0 or bh == 0 or (area / (bw * bh)) < MIN_EXTENT:
            continue
        box = cv2.boxPoints(rect)
        boxes.append({"box": box, "cx": rect[0][0], "area": area})

    boxes.sort(key=lambda item: item["cx"])
    return boxes


def calibrate_scale(boxes, reference_length_cm):
    if not np.isfinite(reference_length_cm) or reference_length_cm <= 0:
        raise ValueError("Panjang referensi harus positif dan finite")
    if not boxes:
        raise ValueError("Kalibrasi gagal: tidak ada kotak terdeteksi")
    largest = max(boxes, key=lambda item: item["area"])
    length_px, _ = box_sides(largest["box"])
    return length_px / reference_length_cm


def draw_frame(frame, boxes, scale, status):
    result = frame.copy()
    for item in boxes:
        box = item["box"]
        cv2.drawContours(result, [box.astype(np.int32)], -1, (0, 255, 0), 2)
        for x, y in box:
            cv2.circle(result, (int(x), int(y)), 4, (0, 0, 255), -1)
        length_px, width_px = box_sides(box)
        mid_x, mid_y = np.mean(box, axis=0).astype(int)
        for prefix, pixels, offset in (("P", length_px, -10), ("L", width_px, 15)):
            cv2.putText(result, f"{prefix}: {pixels / scale:.1f} cm ({int(pixels)}px)",
                        (int(mid_x) - 70, int(mid_y) + offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Kotak: {len(boxes)} | Skala: {scale:.1f} px/cm", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, "[k] Kalibrasi | [+/-] Skala | [q] Keluar", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, status, (15, 85), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, "Kelompok 8 - UTS Machine Vision",
                (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main(argv=None):
    parser, args = parse_args(argv)
    cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            parser.error(f"kamera index {args.camera} tidak dapat dibuka; periksa izin Camera")
        scale = args.scale
        status = "Skala awal/manual; tekan k dengan referensi sebagai kotak terbesar"
        print("k: kalibrasi | +/-: skala manual | q/Esc: keluar")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca frame kamera.")
                return 1
            boxes = detect_boxes(frame)
            cv2.imshow(WINDOW, draw_frame(frame, boxes, scale, status))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return 0
            if key in (ord("k"), ord("K")):
                try:
                    scale = calibrate_scale(boxes, args.reference_length)
                except ValueError as error:
                    status = str(error)
                else:
                    status = "Skala dari kalibrasi referensi; kamera harus tetap diam"
                    print(f"Kalibrasi berhasil: {scale:.2f} px/cm")
            elif key in (ord("+"), ord("=")):
                scale += SCALE_STEP
                status = "Skala diubah manual; validasi dengan penggaris"
            elif key in (ord("-"), ord("_")) and scale > SCALE_STEP:
                scale -= SCALE_STEP
                status = "Skala diubah manual; validasi dengan penggaris"
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
