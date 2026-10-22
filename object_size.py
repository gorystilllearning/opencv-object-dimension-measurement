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


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def filter_outer_boxes(candidates):
    candidates.sort(key=lambda d: d["area"], reverse=True)
    kept = []
    for cand in candidates:
        x, y, w, h = cand["bbox"]
        if not any(
            x >= px - 5 and y >= py - 5 and (x + w) <= (px + pw + 5) and (y + h) <= (py + ph + 5)
            for px, py, pw, ph in [k["bbox"] for k in kept]
        ):
            kept.append(cand)
    return kept


ap = argparse.ArgumentParser(
    description="UTS Machine Vision - Kelompok 8: pengukuran dimensi objek dengan OpenCV",
    epilog="1. Ahmad Bagus Idkholus Surur (23050874166); 2. Khairie Arrantisy (23050874182); "
           "3. Muhammad Ilham Gymnastiar (23050874169); 4. Wahyu Rizki Agung Kurniawan (23050874189)")
ap.add_argument("-s", "--scale", type=float, default=20.0,
                help="skala konversi piksel per cm (default: 20.0 px/cm)")
ap.add_argument("-w", "--width", type=float, default=None,
                help="opsional: kalibrasi skala dari lebar objek referensi paling kiri (cm)")
ap.add_argument("-c", "--camera", type=int, default=0,
                help="index kamera (default: 0)")
args = vars(ap.parse_args())

if args["scale"] <= 0:
    ap.error("--scale harus bernilai positif")

scale = args["scale"]
calibrated = False

cap = cv2.VideoCapture(args["camera"])
if not cap.isOpened():
    ap.error(f"kamera index {args['camera']} tidak dapat dibuka")

print("Program berjalan. Tekan 'q' untuk keluar, '+' atau '-' untuk atur skala.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    raw_cnts = imutils.grab_contours(
        cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

    frame_area = frame.shape[0] * frame.shape[1]
    candidates = []

    for c in raw_cnts:
        area = cv2.contourArea(c)
        if area < 800 or area > 0.85 * frame_area:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
        rect = cv2.minAreaRect(hull)
        bw, bh = rect[1]
        if bw == 0 or bh == 0:
            continue
        extent = area / (bw * bh)
        if 4 <= len(approx) <= 6 and extent >= 0.65:
            box = cv2.boxPoints(rect) if not imutils.is_cv2() else cv2.cv.BoxPoints(rect)
            box = perspective.order_points(np.array(box, dtype="int"))
            bx, by, bw_i, bh_i = cv2.boundingRect(hull)
            candidates.append({
                "box": box,
                "bbox": (bx, by, bw_i, bh_i),
                "area": area,
                "cx": rect[0][0]
            })

    boxes = filter_outer_boxes(candidates)

    if boxes:
        boxes.sort(key=lambda b: b["cx"])

        if args["width"] is not None and not calibrated:
            ref_box = boxes[0]["box"]
            (tl, tr, br, bl) = ref_box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            ref_px = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            if ref_px > 0:
                scale = ref_px / args["width"]
                calibrated = True

        for item in boxes:
            box = item["box"]
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 4, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY)   = midpoint(tl, tr)
            (blbrX, blbrY)   = midpoint(bl, br)
            (tlblX, tlblY)   = midpoint(tl, bl)
            (trbrX, trbrY)   = midpoint(tr, br)

            for pt in [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]:
                cv2.circle(orig, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            dimA = dA / scale
            dimB = dB / scale

            cv2.putText(orig, f"{dimA:.1f} cm", (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(orig, f"{dimB:.1f} cm", (int(trbrX + 10), int(trbrY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(orig, f"Kotak: {len(boxes)} | Skala: {scale:.1f} px/cm", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(orig, "Kelompok 8 - UTS Machine Vision",
                (10, orig.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Object Dimension Measurement", orig)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord("+") or key == ord("="):
        scale += 1.0
    elif key == ord("-") or key == ord("_"):
        if scale > 1.0:
            scale -= 1.0

cap.release()
cv2.destroyAllWindows()