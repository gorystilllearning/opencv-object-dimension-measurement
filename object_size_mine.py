# ==========================================================
# UTS Machine Vision - Pengukuran Dimensi Objek
# Kelompok 8
#   1. Ahmad Bagus Idkholus Surur   (23050874166)
#   2. Khairie Arrantisy            (23050874182)
#   3. Muhammad Ilham Gymnastiar    (23050874169)
#   4. Wahyu Rizki Agung Kurniawan  (23050874189)
# ==========================================================

import argparse
import imutils
import cv2
import numpy as np


def show_image(title, img, destroy=True):
    cv2.imshow(title, img)
    cv2.waitKey(1)
    if destroy:
        cv2.destroyWindow(title)


ap = argparse.ArgumentParser(
    description="UTS Machine Vision - Kelompok 8: pengukuran dimensi objek dengan OpenCV",
    epilog="1. Ahmad Bagus Idkholus Surur (23050874166); 2. Khairie Arrantisy (23050874182); "
           "3. Muhammad Ilham Gymnastiar (23050874169); 4. Wahyu Rizki Agung Kurniawan (23050874189)")
ap.add_argument("-w", "--width", type=float, required=True,
                help="diameter objek referensi lingkaran paling kiri dalam frame (cm)")
ap.add_argument("-c", "--camera", type=int, default=0,
                help="index kamera (default: 0)")
args = vars(ap.parse_args())
if not np.isfinite(args["width"]) or args["width"] <= 0:
    ap.error("--width harus bernilai positif dan finite dalam cm")

cap = cv2.VideoCapture(args["camera"])
if not cap.isOpened():
    ap.error(f"kamera index {args['camera']} tidak dapat dibuka")

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    edged = cv2.Canny(gray, 50, 150)
    show_image("Edged", edged, destroy=False)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    show_image("Morphology", edged, destroy=False)

    raw_cnts = imutils.grab_contours(
        cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

    circles = []
    for c in raw_cnts:
        area = cv2.contourArea(c)
        if area < 350:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        (bx, by, bw, bh) = cv2.boundingRect(c)
        ar = float(bw) / bh if bh > 0 else 0
        ((cx, cy), radius) = cv2.minEnclosingCircle(c)
        circle_area = np.pi * (radius ** 2)
        extent = area / circle_area if circle_area > 0 else 0

        if circ >= 0.60 and extent >= 0.65 and 0.70 <= ar <= 1.40:
            circles.append({
                "cx": cx,
                "cy": cy,
                "radius": radius,
                "d_px": 2.0 * radius
            })

    if circles:
        circles.sort(key=lambda item: item["cx"])
        pixelsPerMetric = circles[0]["d_px"] / args["width"]

        for item in circles:
            cx, cy, r = int(item["cx"]), int(item["cy"]), int(item["radius"])
            dim = item["d_px"] / pixelsPerMetric

            cv2.circle(orig, (cx, cy), r, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(orig, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

            p1 = (cx - r, cy)
            p2 = (cx + r, cy)
            cv2.line(orig, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(orig, p1, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(orig, p2, 4, (0, 255, 255), -1, cv2.LINE_AA)

            label = f"{dim:.1f} cm"
            tx = cx - r
            ty = cy - r - 10 if (cy - r - 10) > 25 else cy + r + 25
            cv2.putText(orig, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(orig, f"Lingkaran: {len(circles)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(orig, "Kelompok 8 - UTS Machine Vision",
                (10, orig.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Object Dimension Measurement", orig)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()