# Ett försök på egen implementering

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import imageio
import glob
import piexif
from scipy.interpolate import interp1d  # Import interpolation function


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

def interpolate_g(g_sparse, Z_sparse, Z_full):
    """
    Interpolate g(Z) for the full set of pixel values Z_full using sparse g(Z) values.
    
    g_sparse: The sparse g(Z) values computed for a subset of Z values (M//10000 points).
    Z_sparse: The corresponding pixel values where g_sparse was computed (M//10000 values).
    Z_full: The full set of pixel values (M values) for which we need to interpolate g(Z).
    """
    # Create an interpolation function
    interpolator = interp1d(Z_sparse, g_sparse, kind='linear', fill_value="extrapolate")
    
    # Interpolate g(Z) for all pixels in Z_full
    g_full = interpolator(Z_full)
    
    return g_full


def compute_matrices(images, exposure_times):
    N = len(images)  # Antalet bilder
    M = images[0].shape[0] * images[0].shape[1] # Antalet pixlar i varje bild (antar att bilderna har samma storlek)

    ones_matrix = -sp.csr_matrix(np.ones((N, M)))

    identity_matrix = sp.eye(N, format='csr')

    tiled_matrix = sp.hstack([identity_matrix] * M, format='csr')

    A = sp.hstack([ones_matrix, tiled_matrix], format='csr')

    # Vektorn b som är ln(exponeringstider)
    ln_delta_t = np.log(exposure_times)
    
    return A, ln_delta_t


def downsample_image(image, r):
    """Scales down an RGB image by averaging pixel values within an r x r block."""
    H, W, C = image.shape  # Height, Width, Channels
    
    # New dimensions after downsampling
    new_H = H // r
    new_W = W // r
    
    # Create an empty array for the downsampled image
    downsampled = np.zeros((new_H, new_W, C), dtype=np.uint8)
    
    # Iterate over blocks of size r x r
    for i in range(new_H):
        for j in range(new_W):
            for c in range(C):  # Iterate over color channels
                block = image[i*r:(i+1)*r, j*r:(j+1)*r, c]  # Extract r x r block
                downsampled[i, j, c] = np.mean(block, dtype=np.float32)  # Compute mean
            
    return downsampled


def compute_weights(Z):
    Zmin = np.min(Z)
    Zmax = np.max(Z)
    #weights = Z * (1 - Z)  # Polynom vikt
    #weights = np.exp(-((Z - 0.5) ** 2) / (2 * 0.1 ** 2)) # Gauss vikt
    weights = np.where(Z <= (Zmin + Zmax) / 2, Z - Zmin, Zmax - Z) # Triangulär vikt
    return weights

def merge_hdr(images, exposure_times, g):
    images = np.stack(images, axis=-1)  # Arrayen är av formen (H, W, C, N) höjd, bredd, färgkanal, bildnummer
    weights = compute_weights(images)
    log_exposures = np.log(exposure_times)
    
    # Tillämpa Debevecs HDR alogritm
    numerator = np.sum(weights * (g - log_exposures.reshape(1,1,1,-1)), axis=-1)
    denominator = np.sum(weights, axis=-1)
    hdr= np.exp(numerator / (denominator + epsilon)) # exponering
    return hdr

def least_squares_solution(A, b):
    x = spla.lsqr(A, b)[0]  
    return x


def save_image(filename, image):
    imageio.imwrite(filename, image)

if __name__ == "__main__":
    image_files = sorted(glob.glob("bilder/*.JPG"))
    
    images, exposure_times = load_exposure_images(image_files)

    images = np.array(images)

    #Zprim = images[:100:,:100:,:, :]

    Zprim = [downsample_image(image, 100) for image in images]

    # Skapa den stora matrisen A och vektorn b
    A, ln_delta_t = compute_matrices(Zprim, exposure_times)

    # Lös systemet genom minsta kvadratmetoden
    x = least_squares_solution(A, ln_delta_t)

    M = Zprim[0].shape[0] * Zprim[0].shape[1]

    # Dela upp resultatet i de olika delarna: e och g(Z)
    e_values = x[:M]  # e är de första N värdena
    g_z_values = x[M:]  # g(Z) är de resterande värdena

    #g_full = interpolate_g(g_z_values, Zprim, images)

    save_image("downsampled.JPG", Zprim[0])

    print(M)
    print("Vektorn e:", e_values)
    print("Vektorn g(Z):", np.shape(Zprim))
