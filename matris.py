# Ett försök på egen implementering

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import imageio
import glob
import piexif

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


def least_squares_solution(A, b):
    x = spla.lsqr(A, b)[0]  
    return x

if __name__ == "__main__":
    image_files = sorted(glob.glob("bilder/*.JPG"))
    
    images, exposure_times = load_exposure_images(image_files)

    # Skapa den stora matrisen A och vektorn b
    A, ln_delta_t = compute_matrices(images, exposure_times)

    # Lös systemet genom minsta kvadratmetoden
    x = least_squares_solution(A, ln_delta_t)

    # Dela upp resultatet i de olika delarna: e och g(Z)
    e_values = x[:len(images)]  # e är de första N värdena
    g_z_values = x[len(images):]  # g(Z) är de resterande värdena

    print("Vektorn e:", e_values)
    print("Vektorn g(Z):", g_z_values)
