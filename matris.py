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
    N = len(Z_sparse)
    g_new = g_sparse.reshape(N, len(g_sparse) // N)
    print(np.shape(g_new))
    print(np.shape(Z_sparse))
    # Create an interpolation function
    g_full = []
    for i in range(N):
        interpolator = interp1d(Z_sparse[i], g_new[i], kind='linear', fill_value="extrapolate")
        
        # Interpolate g(Z) for all pixels in Z_full
        g = interpolator(Z_full)
        g_full.append(g)
    
    return g_full


def create_g_function(g_values, Zprim):
    """
    Creates an interpolation function g that can be applied to a general input Z.
    
    Parameters:
    g_values (np.array): The approximated g function values.
    Zprim (list of np.array): The downsampled pixel intensity values used for approximation.
    
    Returns:
    function: An interpolated function g(Z) that can be applied to general Z values.
    """
    
    # Flatten Zprim to get all sampled intensity values
    Z_flat = np.concatenate([img.flatten() for img in Zprim])
    
    # Ensure uniqueness by sorting and removing duplicates (monotonic increase for interpolation)
    unique_Z = np.unique(Z_flat)
    unique_g = g_values[:len(unique_Z)]  # Match corresponding g values
    
    # Create an interpolation function
    g_interpolated = interp1d(unique_Z, unique_g, kind='linear', fill_value='extrapolate')
    
    return g_interpolated

def approximate_g(images, exposure_times):
    N = len(images)  # Antalet bilder
    M = images[0].shape[0] * images[0].shape[1] # Antalet pixlar i varje bild (antar att bilderna har samma storlek)

    ones_matrix = -sp.csr_matrix(np.ones((N, M)))

    identity_matrix = sp.eye(N, format='csr')

    tiled_matrix = sp.hstack([identity_matrix] * M, format='csr')

    A = sp.hstack([ones_matrix, tiled_matrix], format='csr')

    # Vektorn b som är ln(exponeringstider)
    ln_delta_t = np.log(exposure_times)

    x = spla.lsqr(A, ln_delta_t)[0]  

    g = x[M:]
    
    return g


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
    numerator = np.sum(weights * (g(images) - log_exposures.reshape(1,1,1,-1)), axis=-1)
    denominator = np.sum(weights, axis=-1)
    hdr= np.exp(numerator / (denominator + epsilon)) # exponering
    return hdr

def tone_map(hdr_image):
    # Standard Rec. 709 för att beräkna luminans utifrån RGB värderna
    luminance = 0.2126 * hdr_image[..., 0] + 0.7152 * hdr_image[..., 1] + 0.0722 * hdr_image[..., 2]

    # Beräkna log-medel luminans
    log_average_lum = np.exp(np.mean(np.log(luminance + epsilon)))
    normal_luminance = luminance / log_average_lum

    # Tillämpa Reinhard tone-mapping
    tone_mapped_luminance = normal_luminance / (1 + normal_luminance)

    # Använd tone-mappad luminansen för att skala ner färg kanalerna
    ldr_image = (hdr_image / (luminance[..., None] + epsilon)) * tone_mapped_luminance[..., None]

    ldr_image = np.clip(ldr_image, 0, 1)  # Klipp bort värderna utanför intervallet [0,1]
    ldr_image = (ldr_image * 255).astype(np.uint8) # Konvertera till 8-bitar

    return ldr_image

def save_image(filename, image):
    imageio.imwrite(filename, image)

if __name__ == "__main__":
    image_files = sorted(glob.glob("bilder/*.JPG"))
    
    images, exposure_times = load_exposure_images(image_files)

    images = np.array(images)

    Zprim = [downsample_image(image, 100) for image in images]

    # approximera g
    g_z_values = approximate_g(Zprim, exposure_times)

    M = Zprim[0].shape[0] * Zprim[0].shape[1]

    #g_full = interpolate_g(g_z_values, Zprim, images)

    #save_image("downsampled1.JPG", Zprim[0])
    #save_image("downsampled2.JPG", Zprim[1])
    #save_image("downsampled3.JPG", Zprim[2])

    g = create_g_function(g_z_values, Zprim)

    #hdr_image = merge_hdr(images, exposure_times, g)
    #ldr_image = tone_map(hdr_image)
    #save_image("hdr_image.JPG", ldr_image)
    
