# ==========================================================
# UTS Machine Vision - Pengukuran Dimensi Objek
# Kelompok 8
#   1. Ahmad Bagus Idkholus Surur   (23050874166)
#   2. Khairie Arrantisy            (23050874182)
#   3. Muhammad Ilham Gymnastiar    (23050874169)
#   4. Wahyu Rizki Agung Kurniawan  (23050874189)
# ==========================================================
# USAGE
# python object_size.py --image images/sample_cm.png --width 2.5

from scipy.spatial import distance as dist
from imutils import perspective, contours
import numpy as np, argparse, imutils, cv2

midpoint = lambda ptA, ptB: ((ptA[0]+ptB[0])*0.5, (ptA[1]+ptB[1])*0.5)
pt       = lambda x, y: (int(x), int(y))

ap = argparse.ArgumentParser(
    description="UTS Machine Vision - Kelompok 8: pengukuran dimensi objek dengan OpenCV",
    epilog="1. Ahmad Bagus Idkholus Surur (23050874166); 2. Khairie Arrantisy (23050874182); 3. Muhammad Ilham Gymnastiar (23050874169); 4. Wahyu Rizki Agung Kurniawan (23050874189)")
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in cm)")
args = vars(ap.parse_args())
(not np.isfinite(args["width"]) or args["width"] <= 0) and ap.error("--width must be a positive, finite value in cm")

image = cv2.imread(args["image"])
image is None and ap.error("could not read image: " + args["image"])
gray  = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
edged = cv2.erode(cv2.dilate(cv2.Canny(gray, 50, 100), None, iterations=1), None, iterations=1)

cnts = [c for c in imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)) if cv2.contourArea(c) >= 100]
not cnts and ap.error("no objects found; check contrast and the reference object")
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

for c in cnts:
    orig = image.copy()
    box  = perspective.order_points(np.array(cv2.cv.BoxPoints(cv2.minAreaRect(c)) if imutils.is_cv2() else cv2.boxPoints(cv2.minAreaRect(c)), dtype="int"))
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    [cv2.circle(orig, pt(x, y), 5, (0, 0, 255), -1) for x, y in box]

    (tl, tr, br, bl)   = box
    (tltrX, tltrY)     = midpoint(tl, tr)
    (blbrX, blbrY)     = midpoint(bl, br)
    (tlblX, tlblY)     = midpoint(tl, bl)
    (trbrX, trbrY)     = midpoint(tr, br)

    [cv2.circle(orig, pt(*m), 5, (255, 0, 0), -1) for m in [(tltrX,tltrY),(blbrX,blbrY),(tlblX,tlblY),(trbrX,trbrY)]]
    cv2.line(orig, pt(tltrX, tltrY), pt(blbrX, blbrY), (255, 0, 255), 2)
    cv2.line(orig, pt(tlblX, tlblY), pt(trbrX, trbrY), (255, 0, 255), 2)

    dA, dB           = dist.euclidean((tltrX,tltrY),(blbrX,blbrY)), dist.euclidean((tlblX,tlblY),(trbrX,trbrY))
    pixelsPerMetric  = pixelsPerMetric or (dB / args["width"])
    dimA, dimB       = dA / pixelsPerMetric, dB / pixelsPerMetric

    cv2.putText(orig, f"{dimA:.1f} cm", pt(tltrX-15, tltrY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(orig, f"{dimB:.1f} cm", pt(trbrX+10, trbrY),    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(orig, "Kelompok 8 - UTS Machine Vision", (10, orig.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow("Image", orig); cv2.waitKey(0)