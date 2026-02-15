# =====================================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Unit: Image Restoration
# Assignment: Noise Modeling and Image Restoration
# Date: 12-Feb-2026
# =====================================================

import cv2
import numpy as np
import os


# -----------------------------------------------------
# Welcome Message
# -----------------------------------------------------
def welcome():
    print("=" * 60)
    print(" IMAGE RESTORATION SYSTEM ")
    print("=" * 60)
    print("Task 1 & Task 2: Image Loading + Noise Modeling\n")


# -----------------------------------------------------
# Load and Preprocess Image
# -----------------------------------------------------
def load_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Error loading:", path)
        return None

    # Resize to standard size
    img = cv2.resize(img, (512, 512))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


# -----------------------------------------------------
# Add Gaussian Noise
# -----------------------------------------------------
def add_gaussian_noise(image):

    mean = 0
    std = 25

    gaussian = np.random.normal(mean, std, image.shape)

    noisy = image + gaussian

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


# -----------------------------------------------------
# Add Salt and Pepper Noise
# -----------------------------------------------------
def add_salt_pepper_noise(image):

    noisy = image.copy()

    prob = 0.02  # 2% noise

    # Salt (white pixels)
    salt = np.random.rand(*image.shape) < prob
    noisy[salt] = 255

    # Pepper (black pixels)
    pepper = np.random.rand(*image.shape) < prob
    noisy[pepper] = 0

    return noisy


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
        print("No images found in images folder.")
        return

    for file in files:

        print("Processing:", file)

        path = os.path.join(image_folder, file)

        original_gray = load_image(path)

        if original_gray is None:
            continue

        # ---------------- Add Noise ----------------
        gaussian_noisy = add_gaussian_noise(original_gray)
        salt_pepper_noisy = add_salt_pepper_noise(original_gray)

        # ---------------- Display ----------------
        cv2.imshow("Original Grayscale", original_gray)
        cv2.imshow("Gaussian Noise", gaussian_noisy)
        cv2.imshow("Salt & Pepper Noise", salt_pepper_noisy)

        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -----------------------------------------------------
# Program Entry
# -----------------------------------------------------
if __name__ == "__main__":
    main()
