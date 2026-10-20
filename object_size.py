# ==========================================================
# UTS Machine Vision - Pengukuran Dimensi Objek
# Kelompok 8
#   1. Ahmad Bagus Idkholus Surur   (23050874166)
#   2. Khairie Arrantisy            (23050874182)
#   3. Muhammad Ilham Gymnastiar    (23050874169)
#   4. Wahyu Rizki Agung Kurniawan  (23050874189)
# ==========================================================

from scipy.spatial import distance as dist
from imutils import perspective, contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


ap = argparse.ArgumentParser(
    description="UTS Machine Vision - Kelompok 8: pengukuran dimensi objek dengan OpenCV",
    epilog="1. Ahmad Bagus Idkholus Surur (23050874166); 2. Khairie Arrantisy (23050874182); "
           "3. Muhammad Ilham Gymnastiar (23050874169); 4. Wahyu Rizki Agung Kurniawan (23050874189)")
ap.add_argument("-w", "--width", type=float, required=True,
                help="lebar objek referensi paling kiri dalam frame (cm)")
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

    orig  = frame.copy()
    gray  = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    raw_cnts = imutils.grab_contours(
        cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    cnts = [c for c in raw_cnts if cv2.contourArea(c) >= 100]

    if cnts:
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        for c in cnts:
            box = cv2.boxPoints(cv2.minAreaRect(c)) if not imutils.is_cv2() else cv2.cv.BoxPoints(cv2.minAreaRect(c))
            box = perspective.order_points(np.array(box, dtype="int"))
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY)   = midpoint(tl, tr)
            (blbrX, blbrY)   = midpoint(bl, br)
            (tlblX, tlblY)   = midpoint(tl, bl)
            (trbrX, trbrY)   = midpoint(tr, br)

            for pt in [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]:
                cv2.circle(orig, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / args["width"]

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            cv2.putText(orig, f"{dimA:.1f} cm", (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(orig, f"{dimB:.1f} cm", (int(trbrX + 10), int(trbrY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(orig, "Kelompok 8 - UTS Machine Vision",
                (10, orig.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Object Dimension Measurement", orig)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()