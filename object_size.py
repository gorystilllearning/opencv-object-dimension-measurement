# ==========================================================
# UTS Machine Vision - Pengukuran Dimensi Objek
# Kelompok 8
#   1. Ahmad Bagus Idkholus Surur   (23050874166)
#   2. Khairie Arrantisy            (23050874182)
#   3. Muhammad Ilham Gymnastiar    (23050874169)
#   4. Wahyu Rizki Agung Kurniawan  (23050874189)
# ==========================================================

from scipy.spatial import distance as dist
from imutils import perspective
import argparse
import imutils
import cv2
import numpy as np


def is_rectangular(pts):
    pts = pts.reshape(4, 2)
    for i in range(4):
        p0, p1, p2 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_th = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        if cos_th > 0.35:
            return False
    return True


ap = argparse.ArgumentParser(
    description="UTS Machine Vision - Kelompok 8: pengukuran dimensi objek dengan OpenCV",
    epilog="1. Ahmad Bagus Idkholus Surur (23050874166); 2. Khairie Arrantisy (23050874182); "
           "3. Muhammad Ilham Gymnastiar (23050874169); 4. Wahyu Rizki Agung Kurniawan (23050874189)")
ap.add_argument("-s", "--scale", type=float, default=20.0,
                help="skala konversi piksel per cm (default: 20.0 px/cm)")
ap.add_argument("-r", "--ref", type=float, default=8.5,
                help="panjang acuan kalibrasi dalam cm (default: 8.5 cm untuk kartu ID/KTP)")
ap.add_argument("-w", "--width", type=float, default=None,
                help="opsional: kompatibilitas mundur lebar acuan kalibrasi (cm)")
ap.add_argument("-c", "--camera", type=int, default=0,
                help="index kamera (default: 0)")
args = vars(ap.parse_args())

if args["scale"] <= 0:
    ap.error("--scale harus bernilai positif")

scale = args["scale"]
ref_len = args["width"] if args["width"] is not None else args["ref"]

cap = cv2.VideoCapture(args["camera"])
if not cap.isOpened():
    ap.error(f"kamera index {args['camera']} tidak dapat dibuka")

print("Program berjalan. Tekan 'k' kalibrasi, '+/-' atur skala, 'q' keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    raw_cnts = imutils.grab_contours(
        cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

    frame_area = frame.shape[0] * frame.shape[1]
    boxes = []

    for c in raw_cnts:
        area = cv2.contourArea(c)
        if area < 1500 or area > 0.85 * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if not is_rectangular(approx):
            continue
        rect = cv2.minAreaRect(c)
        bw, bh = rect[1]
        if bw == 0 or bh == 0:
            continue
        extent = area / (bw * bh)
        if extent < 0.70:
            continue
        box = cv2.boxPoints(rect) if not imutils.is_cv2() else cv2.cv.BoxPoints(rect)
        box = perspective.order_points(np.array(box, dtype="int"))
        boxes.append({"box": box, "cx": rect[0][0], "area": area})

    if boxes:
        boxes.sort(key=lambda b: b["cx"])

        for item in boxes:
            box = item["box"]
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 4, (0, 0, 255), -1)

            s1 = dist.euclidean(box[0], box[1])
            s2 = dist.euclidean(box[1], box[2])
            panjang_px = max(s1, s2)
            lebar_px = min(s1, s2)

            panjang_cm = panjang_px / scale
            lebar_cm = lebar_px / scale

            mid_x = int(np.mean(box[:, 0]))
            mid_y = int(np.mean(box[:, 1]))

            label_p = f"P: {panjang_cm:.1f} cm ({int(panjang_px)}px)"
            label_l = f"L: {lebar_cm:.1f} cm ({int(lebar_px)}px)"

            cv2.putText(orig, label_p, (mid_x - 70, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(orig, label_l, (mid_x - 70, mid_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(orig, f"Kotak: {len(boxes)} | Skala: {scale:.1f} px/cm", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(orig, "[k] Kalibrasi | [+/-] Skala | [q] Keluar", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(orig, "Kelompok 8 - UTS Machine Vision",
                (10, orig.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Object Dimension Measurement", orig)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord("k") and boxes:
        largest = max(boxes, key=lambda b: b["area"])["box"]
        s1 = dist.euclidean(largest[0], largest[1])
        s2 = dist.euclidean(largest[1], largest[2])
        longest_px = max(s1, s2)
        if longest_px > 0 and ref_len > 0:
            scale = longest_px / ref_len
            print(f"Kalibrasi berhasil! Skala baru: {scale:.2f} px/cm (berdasarkan {ref_len} cm)")
    elif key == ord("+") or key == ord("="):
        scale += 0.5
    elif key == ord("-") or key == ord("_"):
        if scale > 1.0:
            scale -= 0.5

cap.release()
cv2.destroyAllWindows()