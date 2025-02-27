# Ett försök på egen implementering

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import imageio
import glob
import piexif
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Democratic_tonemap import tonemap
import cv2

def align_images(images):
    N, H, W, C = images.shape  # Extract dimensions
    ref_idx = 0  # Choose the first image as the reference
    ref_image = images[ref_idx]
    aligned_images = [ref_image]

    # Convert reference to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    # Use SIFT instead of ORB for better feature detection
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(ref_gray, None)

    for i in range(1, N):
        img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # Apply Adaptive Histogram Equalization (CLAHE) to improve feature detection in dark images
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        keypoints2, descriptors2 = sift.detectAndCompute(img_gray, None)

        if descriptors2 is None or len(keypoints2) < 10:  # Skip alignment if too few features
            print(f"Skipping alignment for image {i} due to low feature detection.")
            aligned_images.append(images[i])
            continue

        # Use FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe’s ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < 10:  # Skip alignment if not enough good matches
            print(f"Skipping alignment for image {i} due to insufficient good matches.")
            aligned_images.append(images[i])
            continue

        # Extract matched keypoints
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute Affine Transformation instead of Homography (prevents stretching)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if M is None:  # Skip if transformation fails
            print(f"Skipping alignment for image {i} due to transformation failure.")
            aligned_images.append(images[i])
            continue

        # Warp image using Affine transformation (prevents stretching)
        aligned = cv2.warpAffine(images[i], M, (W, H))

        aligned_images.append(aligned)

    return np.array(aligned_images)


epsilon = 1e-6

def load_exposure_images(image_files):
    images = []
    exposure_times = []
    
    for img_path in image_files:
        image = imageio.v2.imread(img_path).astype(np.uint8)
        images.append(image)
        
        exif_data = piexif.load(img_path)
        exposure_time = exif_data['Exif'][piexif.ExifIFD.ExposureTime]
        exposure_times.append(exposure_time[0] / exposure_time[1])
    
    exposure_times = np.array(exposure_times, dtype=np.float32)
    return images, exposure_times


def create_g_function(g_values, Zprim, delta_t):
    N = len(g_values) // len(Zprim)
    g_z = g_values[:N]

    print(np.shape(g_z))

    img = Zprim[0]
    first_channel = img[:, 1].flatten()

    # Sortera datan för "garanterad" monoton funktion
    first_channel_sorted = np.sort(first_channel) + epsilon
    g_z_sorted = np.sort(g_z) 

    # Interpolera för att få en fullständig funktion
    g_interpolated = interp1d(first_channel_sorted, g_z_sorted, kind='linear', fill_value='extrapolate')

    return g_interpolated


def approximate_g(images, exposure_times, M, lambda_smooth=1000):
    N = len(images)  # Antalet bilder
    #M = images[0].shape[0] * images[0].shape[1] # Antalet pixlar i varje bild (antar att bilderna har samma storlek)

    ones_matrix = -sp.csr_matrix(np.ones((N, M)))

    # Create the second block (N x NM) with structured ones
    second_block = sp.lil_matrix((N, N * M))
    tri_values = np.concatenate([np.linspace(0, 1, M//2), np.linspace(1, 0, M - M//2)])

    for i in range(N):
        second_block[i, i * M : (i + 1) * M] = tri_values

    # Convert second block to csr format
    second_block = second_block.tocsr()

    A = sp.hstack([ones_matrix, second_block], format='csr')

    # Vektorn b som är ln(exponeringstider)
    ln_delta_t = M * np.log(exposure_times)


    # # Andra derivatan matrisen
    # zero_matrix = sp.csr_matrix((N,M))
    # neg_2I = -2 * identity_matrix

    # blocks = []
    
    # # Cyclysikt identitetsmatriser
    # for _ in range(M // 3):
    #     blocks.extend([identity_matrix, neg_2I, identity_matrix])
    
    # D = sp.hstack(blocks, format='csr')
    # A_reg = sp.hstack([zero_matrix, lambda_smooth * D], format='csr')
    
    # A_combined = A + A_reg #sp.vstack([A, A_reg], format='csr')
    # b_combined = np.hstack([ln_delta_t, np.zeros(N)])

    # # Add constraint g(128) = 0
    # constraint = sp.lil_matrix((N, A_combined.shape[1]))
    # for i in range(N):
    #     constraint[i, M + M //2 + i] = 1

    # A_combined = sp.vstack([A_combined, constraint], format='csr')
    # b_combined = np.hstack([ln_delta_t, np.ones(N)])
    # print(np.shape(A_combined))
    # print(np.shape(b_combined))
    # lös minsta kvadrat problemet
    x = spla.lsqr(A, ln_delta_t)[0]  

    g = x[M:]
    lE = x[:M]
    
    return g, lE


# def downsample_image(image, r):
#     H, W, C = image.shape  # Höjd, Bredd, Färgkanal
    
#     # Nya höjd och bredd dimensioner
#     new_H = H // r
#     new_W = W // r
    
#     downsampled = np.zeros((new_H, new_W, C), dtype=np.uint8)
    
#     for i in range(new_H):
#         for j in range(new_W):
#             for c in range(C): # iteration över varje färgkanal
#                 # Beräkna medelvärdet för ett r x r block
#                 block = image[i*r:(i+1)*r, j*r:(j+1)*r, c] 
#                 downsampled[i, j, c] = np.mean(block, dtype=np.float32)
            
#     return downsampled


def compute_weights(Z):
    Zmin = np.min(Z)
    Zmax = np.max(Z)
    #weights = Z * (1 - Z)  # Polynom vikt
    #weights = np.exp(-((Z - 0.5) ** 2) / (2 * 0.1 ** 2)) # Gauss vikt
    weights = np.where(Z <= (Zmin + Zmax) / 2, Z - Zmin, Z) # Triangulär vikt

    return weights

def compute_image_weights(images):
    image_weights  = []
    for i in range(len(images)):
        Z = images[i]
        weights = compute_weights(Z)
        image_weights.append(weights)
    return image_weights


def merge_hdr(images, exposure_times, g):
    # Arrayen är av formen (N, H, W, C) bildnummer, höjd, bredd, färgkanal
    image_weights = compute_image_weights(images)
    log_exposures = np.log(exposure_times)
    
    # Tillämpa Debevecs HDR alogritm
    numerator = np.sum(image_weights * np.exp((g(images) - log_exposures[:, None, None, None])), axis=0)
    #denominator = np.sum(image_weights, axis=0)

    #hdr= np.exp(numerator / (denominator + epsilon))
    return numerator

def tone_map(hdr_image):
    # Standard Rec. 709 för att beräkna luminans utifrån RGB värderna
    luminance = 0.2126 * hdr_image[..., 0] + 0.7152 * hdr_image[..., 1] + 0.0722 * hdr_image[..., 2]

    # Antal pixlar
    N = luminance.size
    
    # Beräkna log-medel luminans
    log_average_lum = np.exp(np.sum(np.log(luminance + epsilon)) / N)

    normal_luminance = luminance / log_average_lum

    # Tillämpa Reinhard tone-mapping
    tone_mapped_luminance = normal_luminance / (1 + normal_luminance)
    
    # Använd tone-mappad luminansen för att skala ner färg kanalerna
    ldr_image = (hdr_image / (luminance[..., None] + epsilon)) * tone_mapped_luminance[..., None]

    ldr_image = np.clip(ldr_image, 0, 1)  # Klipp bort värderna utanför intervallet [0,1]
    ldr_image = (ldr_image * 255).astype(np.uint8) # Konvertera till 8-bitar

    return ldr_image

def democratic_tonemapping(image, percentile=90):
    """
    Apply democratic tone mapping to an image.
    
    Parameters:
        image (numpy.ndarray): Input HDR image (float32 or float64, range 0-1 or 0-255).
        percentile (float): The percentile used for normalization (default: 90).
        
    Returns:
        numpy.ndarray: Tone-mapped image (uint8, range 0-255).
    """
    # Ensure the image is in float format
    image = image.astype(np.float32)
    
    # Compute the percentile intensity per channel
    percentiles = np.percentile(image, percentile, axis=(0, 1))
    
    # Normalize each channel separately
    tone_mapped = image / (percentiles + 1e-6)  # Avoid division by zero
    tone_mapped = np.clip(tone_mapped, 0, 1)  # Clip values to [0,1]
    
    # Convert to 8-bit format
    tone_mapped = (tone_mapped * 255).astype(np.uint8)
    
    return tone_mapped

def save_image(filename, image):
    imageio.imwrite(filename, image)

def rand_pixels(images, k=900):
    Zprim = []

    for image in images:
        h, w, _ = image.shape  # Bild dimensioner
        random_rows = np.random.randint(0, h, k)
        random_cols = np.random.randint(0, w, k)
        sampled_pixels = image[random_rows, random_cols, :]
        Zprim.append(sampled_pixels)

    return Zprim

if __name__ == "__main__":
    image_files = sorted(glob.glob("bilder/*.JPG"))
    
    images, exposure_times = load_exposure_images(image_files)

    images = align_images(np.array(images))

    save_image("downsampled1.JPG", images[2])

    #Zprim image[:, ::100,::100,:]

    Zprim = rand_pixels(images)
    print(np.shape(Zprim))

    # approximera g
    g_z_values, lE = approximate_g(Zprim, exposure_times, M=900)

    M = Zprim[0].shape[0] * Zprim[0].shape[1]

    #save_image("downsampled1.JPG", Zprim[0])
    #save_image("downsampled2.JPG", Zprim[1])
    #save_image("downsampled3.JPG", Zprim[2])

    g = create_g_function(g_z_values, Zprim, exposure_times[0])


    x = np.linspace(0, 255, 100)
    y = np.exp(g(x)) / exposure_times[0]

    plt.plot(y, x, label='CRF')
    plt.xlabel("Luminans $e_i$")
    plt.ylabel("Pixelvärde $Z=CRF(e_i \cdot \Delta t)$")
    plt.legend()
    plt.grid(True)
    plt.show()

    hdr_image = merge_hdr(images, exposure_times, g)
    ldr_image = tone_map(hdr_image)
    save_image("hdr_image.JPG", ldr_image)
    
