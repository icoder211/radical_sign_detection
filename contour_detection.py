import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread("images/pg1_image_radical.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(im, (1, 1), 0)#[cv2.GaussianBlur(im, (i, i), 0) for i in range(10, 100, 20)]
im = blur
ret, th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# plt.imshow(th, 'gray')
# plt.show()
cv2.imshow('in', th)
cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

print(len(cnts))
for c in cnts:
    print (c)
    cv2.drawContours(th, [c], 0, (0, 255, 0), thickness=5)
    cv2.imshow('cnts', th)
    # plt.imshow(th)
    # plt.show()
    cv2.waitKey(0)

cv2.destroyAllWindows()

# some of these c's could represent radicals
