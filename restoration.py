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
import math


# -----------------------------------------------------
# Welcome Message
# -----------------------------------------------------
def welcome():
    print("=" * 60)
    print(" IMAGE RESTORATION SYSTEM ")
    print("=" * 60)
    print("Task 1-4: Noise Modeling + Restoration + Evaluation\n")


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
# Performance Evaluation (Task 4)
# -----------------------------------------------------
def calculate_mse(original, restored):

    error = np.mean((original - restored) ** 2)
    return error


def calculate_psnr(original, restored):

    mse = calculate_mse(original, restored)

    if mse == 0:
        return float("inf")

    psnr = 10 * math.log10((255 ** 2) / mse)
    return psnr


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

        print("\nProcessing:", file)

        path = os.path.join(image_folder, file)

        # -------- Task 1 --------
        original = load_image(path)

        if original is None:
            continue

        # -------- Task 2 --------
        noisy_gaussian = add_gaussian_noise(original)
        noisy_sp = add_salt_pepper(original)

        # -------- Task 3 --------
        mean_g = mean_filter(noisy_gaussian)
        median_g = median_filter(noisy_gaussian)
        gauss_g = gaussian_filter(noisy_gaussian)

        mean_sp = mean_filter(noisy_sp)
        median_sp = median_filter(noisy_sp)
        gauss_sp = gaussian_filter(noisy_sp)

        # -------- Task 4: Evaluation --------
        print("\n--- Gaussian Noise Restoration Metrics ---")
        print("Mean Filter -> MSE:",
              calculate_mse(original, mean_g),
              "PSNR:",
              calculate_psnr(original, mean_g))

        print("Median Filter -> MSE:",
              calculate_mse(original, median_g),
              "PSNR:",
              calculate_psnr(original, median_g))

        print("Gaussian Filter -> MSE:",
              calculate_mse(original, gauss_g),
              "PSNR:",
              calculate_psnr(original, gauss_g))

        print("\n--- Salt & Pepper Noise Restoration Metrics ---")
        print("Mean Filter -> MSE:",
              calculate_mse(original, mean_sp),
              "PSNR:",
              calculate_psnr(original, mean_sp))

        print("Median Filter -> MSE:",
              calculate_mse(original, median_sp),
              "PSNR:",
              calculate_psnr(original, median_sp))

        print("Gaussian Filter -> MSE:",
              calculate_mse(original, gauss_sp),
              "PSNR:",
              calculate_psnr(original, gauss_sp))

        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -----------------------------------------------------
# Program Entry
# -----------------------------------------------------
if __name__ == "__main__":
    main()
