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
import numpy as np


# -----------------------------------------------------
# Welcome Message
# -----------------------------------------------------
def welcome():
    print("=" * 60)
    print(" IMAGE RESTORATION SYSTEM ")
    print("=" * 60)
    print("Task 1, 2 & 3: Noise Modeling + Restoration Filters\n")


# -----------------------------------------------------
# Load Image (Task 1)
# -----------------------------------------------------
def load_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Error loading image:", path)
        return None

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


# -----------------------------------------------------
# Noise Models (Task 2)
# -----------------------------------------------------
def add_gaussian_noise(image):

    mean = 0
    std = 25
    gaussian = np.random.normal(mean, std, image.shape)
    noisy = image + gaussian

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(image):

    noisy = image.copy()
    prob = 0.02

    salt = np.random.rand(*image.shape) < prob
    noisy[salt] = 255

    pepper = np.random.rand(*image.shape) < prob
    noisy[pepper] = 0

    return noisy


# -----------------------------------------------------
# Restoration Filters (Task 3)
# -----------------------------------------------------
def mean_filter(image):
    return cv2.blur(image, (5, 5))


def median_filter(image):
    return cv2.medianBlur(image, 5)


def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# -----------------------------------------------------
# Main Function
# -----------------------------------------------------
def main():

    welcome()

    image_folder = "images"

    if not os.path.exists(image_folder):
        print("Images folder not found!")
        return

    files = os.listdir(image_folder)

    if len(files) == 0:
        print("No images found!")
        return

    for file in files:

        print("Processing:", file)

        path = os.path.join(image_folder, file)

        # -------- Task 1 --------
        original = load_image(path)

        if original is None:
            continue

        # -------- Task 2 --------
        noisy_gaussian = add_gaussian_noise(original)
        noisy_sp = add_salt_pepper(original)

        # -------- Task 3 --------
        # For Gaussian Noise
        mean_g = mean_filter(noisy_gaussian)
        median_g = median_filter(noisy_gaussian)
        gauss_g = gaussian_filter(noisy_gaussian)

        # For Salt & Pepper Noise
        mean_sp = mean_filter(noisy_sp)
        median_sp = median_filter(noisy_sp)
        gauss_sp = gaussian_filter(noisy_sp)

        # -------- Display --------
        cv2.imshow("Original", original)

        cv2.imshow("Gaussian Noise", noisy_gaussian)
        cv2.imshow("Mean Filter (Gaussian)", mean_g)
        cv2.imshow("Median Filter (Gaussian)", median_g)
        cv2.imshow("Gaussian Filter (Gaussian)", gauss_g)

        cv2.imshow("Salt & Pepper Noise", noisy_sp)
        cv2.imshow("Mean Filter (S&P)", mean_sp)
        cv2.imshow("Median Filter (S&P)", median_sp)
        cv2.imshow("Gaussian Filter (S&P)", gauss_sp)

        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -----------------------------------------------------
# Program Entry
# -----------------------------------------------------
if __name__ == "__main__":
    main()
