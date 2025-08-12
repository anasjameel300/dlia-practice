import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
from scipy.ndimage import generic_filter
from skimage.color import rgb2gray
from skimage import color

def main():
    print("=== Feature Extraction Operations ===\n")
    
    # Load image
    img = cv2.imread('sample_img.png')
    if img is None:
        raise FileNotFoundError("Image not found at the specified path. Please check the file location and name.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = color.rgb2gray(img)
    img_gray_uint8 = (img_gray * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. HOG Features
    print("\n1. Computing HOG Features")
    features, hog_image = hog(gray,
                            orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            visualize=True,
                            block_norm='L2-Hys')
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)  # Show color image instead of grayscale
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('HOG Features')
    plt.imshow(hog_image, cmap='gray')
    plt.axis('off')
    plt.show()
    print("HOG feature vector length:", len(features))

    # 2. GLCM Features
    print("\n2. Computing GLCM Features")
    glcm = graycomatrix(img_gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    print("GLCM Features:")
    print(f"  Contrast: {contrast}")
    print(f"  Correlation: {correlation}")
    print(f"  Energy: {energy}")
    print(f"  Homogeneity: {homogeneity}")

    # 3. Local Binary Pattern
    print("\n3. Computing Local Binary Pattern")
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(lbp, cmap='gray')
    plt.title('LBP Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("LBP Feature Vector length:", len(hist))

    # 4. Local Contrast Measure
    print("\n4. Computing Local Contrast Measure")
    def local_contrast(window):
        return np.std(window)
    
    lcm = generic_filter(img_gray.astype(float), local_contrast, size=3)
    plt.figure()
    plt.imshow(lcm, cmap='hot')
    plt.title("Local Contrast (LCM)")
    plt.axis("off")
    plt.colorbar()
    plt.show()

    lcm_mean = np.mean(lcm)
    lcm_std = np.std(lcm)
    print(f"LCM Mean: {lcm_mean}")
    print(f"LCM Std Dev: {lcm_std}")

    # 5. Hu Moments
    print("\n5. Computing Hu Moments")
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    plt.figure()
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title('Contours on Image (input to Hu Moments)')
    plt.axis('off')
    plt.show()

    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    print("Hu Moments values:")
    for i, val in enumerate(hu_moments, 1):
        print(f"Hu Moment {i}: {val}")

    # 6. Color Histograms
    print("\n6. Computing Color Histograms")
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()

    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(hist_r, color='r', label='Red Channel')
    plt.plot(hist_g, color='g', label='Green Channel')
    plt.plot(hist_b, color='b', label='Blue Channel')
    plt.title("Normalized Color Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. SIFT Features
    print("\n7. Computing SIFT Features")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print(f"Number of keypoints: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")

    img_sift = cv2.drawKeypoints(img_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure()
    plt.imshow(img_sift)
    plt.title('SIFT Keypoints')
    plt.axis('off')
    plt.show()

    # 8. ORB Features
    print("\n8. Computing ORB Features")
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    print(f"Number of ORB keypoints: {len(keypoints)}")

    img_orb = cv2.drawKeypoints(img_rgb, keypoints, None, color=(0,255,0), flags=0)
    plt.figure()
    plt.imshow(img_orb)
    plt.title('ORB Keypoints')
    plt.axis('off')
    plt.show()

    # 9. Gabor Filters
    print("\n9. Applying Gabor Filters")
    def build_gabor_kernels():
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            for sigma in (1, 3):
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
        return kernels

    kernels = build_gabor_kernels()
    filtered_images = [cv2.filter2D(gray, cv2.CV_8UC3, k) for k in kernels]

    plt.figure(figsize=(12, 6))
    for i, filtered in enumerate(filtered_images[:4], 1):
        plt.subplot(1, 4, i)
        plt.imshow(filtered, cmap='gray')
        plt.title(f'Gabor Filter {i}')
        plt.axis('off')
    plt.show()

    # 10. Edge Detection
    print("\n10. Performing Edge Detection")
    edges = cv2.Canny(gray, 100, 200)
    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    plt.show()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(sobelx), cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(sobely), cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')
    plt.show()

    print("\n✓ All feature extraction operations completed!")
    print("=== Feature Extraction Completed ===")

if __name__ == "__main__":
    main()
