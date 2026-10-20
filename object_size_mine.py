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
import argparse
import numpy as np
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def show_image(title, img, destroy=True):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()


ap = argparse.ArgumentParser(
    description="UTS Machine Vision - Kelompok 8: pengukuran dimensi objek dengan OpenCV",
    epilog="1. Ahmad Bagus Idkholus Surur (23050874166); 2. Khairie Arrantisy (23050874182); "
           "3. Muhammad Ilham Gymnastiar (23050874169); 4. Wahyu Rizki Agung Kurniawan (23050874189)")
ap.add_argument("-i", "--image",  required=True, help="path to the input image")
ap.add_argument("-w", "--width",  type=float, required=True,
                help="width of the left-most object in the image (in cm)")
args = vars(ap.parse_args())
if not np.isfinite(args["width"]) or args["width"] <= 0:
    ap.error("--width must be a positive, finite value in cm")

image = cv2.imread(args["image"])
if image is None:
    ap.error("could not read image: " + args["image"])

gray  = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
edged = cv2.Canny(gray, 50, 100)
show_image("Edged", edged, destroy=False)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
show_image("erode and dilate", edged)

cnts = imutils.grab_contours(
    cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
print("Total number of contours are:", len(cnts))
cnts = [c for c in cnts if cv2.contourArea(c) >= 100]
if not cnts:
    ap.error("no objects found; check contrast and the reference object")

(cnts, _)      = contours.sort_contours(cnts)
pixelPerMetric = None

for count, c in enumerate(cnts, 1):
    orig = image.copy()
    box  = cv2.boxPoints(cv2.minAreaRect(c)) if not imutils.is_cv2() else cv2.cv.BoxPoints(cv2.minAreaRect(c))
    box  = perspective.order_points(np.array(box, dtype="int"))
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
    if pixelPerMetric is None:
        pixelPerMetric = dB / args["width"]

    dimA = dA / pixelPerMetric
    dimB = dB / pixelPerMetric

    cv2.putText(orig, f"{dimA:.1f} cm", (int(tltrX - 15), int(tltrY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig, f"{dimB:.1f} cm", (int(trbrX + 10), int(trbrY)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig, "Kelompok 8 - UTS Machine Vision",
                (10, orig.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Image", orig)
    cv2.waitKey(0)

print("Total contours processed:", count)