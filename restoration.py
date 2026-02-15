# =====================================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Unit: Image Restoration
# Assignment: Noise Modeling and Image Restoration
# Date: 12-Feb-2026
# =====================================================

import cv2
import os


def welcome():
    print("="*60)
    print(" IMAGE RESTORATION SYSTEM ")
    print("="*60)


def load_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Error loading image.")
        return None

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


def main():

    welcome()

    files = os.listdir("images")

    for file in files:

        path = os.path.join("images", file)

        gray = load_image(path)

        if gray is None:
            continue

        cv2.imshow("Original Grayscale", gray)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
