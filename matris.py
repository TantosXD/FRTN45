# Ett försök på egen implementering

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import imageio
import glob
import piexif
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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


def create_g_function(g_values, Zprim):
    N = len(g_values) // len(Zprim)
    g_z = g_values[:N]

    img = Zprim[0]
    first_channel = img[:, :, 1].flatten()

    # Sortera datan för "garanterad" monoton funktion
    first_channel_sorted = np.sort(first_channel)
    g_z_sorted = np.sort(g_z)

    # Plotta funktionen
    plt.plot(first_channel_sorted, np.exp(g_z_sorted), label='CRF(x)')
    plt.xlabel('x')
    plt.ylabel('CRF(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Interpolera för att få en ordentlig funktion
    g_interpolated = interp1d(first_channel_sorted, g_z_sorted, kind='linear', fill_value='extrapolate')

    return g_interpolated


def approximate_g(images, exposure_times, lambda_smooth=10000):
    N = len(images)  # Antalet bilder
    M = images[0].shape[0] * images[0].shape[1] # Antalet pixlar i varje bild (antar att bilderna har samma storlek)

    ones_matrix = -sp.csr_matrix(np.ones((N, M)))

    identity_matrix = sp.eye(N, format='csr')

    tiled_matrix = sp.hstack([identity_matrix] * M, format='csr')

    A = sp.hstack([ones_matrix, tiled_matrix], format='csr')

    # Vektorn b som är ln(exponeringstider)
    ln_delta_t = M * np.log(exposure_times)


    # Andra derivatan matrisen
    zero_matrix = sp.csr_matrix((N,M))
    neg_2I = -2 * identity_matrix

    blocks = []
    
    # Cyclysikt identitetsmatriser
    for _ in range(M // 3):
        blocks.extend([identity_matrix, neg_2I, identity_matrix])
    
    D = sp.hstack(blocks, format='csr')
    A_reg = sp.hstack([zero_matrix, lambda_smooth * D], format='csr')
    
    A_combined = sp.vstack([A, A_reg], format='csr')
    b_combined = np.hstack([ln_delta_t, np.zeros(N)])

    # lös minsta kvadrat problemet
    x = spla.lsqr(A_combined, b_combined)[0]  

    g = x[M:]
    
    return g


def downsample_image(image, r):
    """Skala ned en RGB bild genom at ta medel pixel värden inom ett r x r block."""
    H, W, C = image.shape  # Höjd, Bredd, Färgkanal
    
    # Nya höjd och bredd dimensioner
    new_H = H // r
    new_W = W // r
    
    downsampled = np.zeros((new_H, new_W, C), dtype=np.uint8)
    
    for i in range(new_H):
        for j in range(new_W):
            for c in range(C): # iteration över varje färgkanal
                # Beräkna medelvärdet för ett r x r block
                block = image[i*r:(i+1)*r, j*r:(j+1)*r, c] 
                downsampled[i, j, c] = np.mean(block, dtype=np.float32)
            
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

    #save_image("downsampled1.JPG", Zprim[0])
    #save_image("downsampled2.JPG", Zprim[1])
    #save_image("downsampled3.JPG", Zprim[2])

    g = create_g_function(g_z_values, Zprim)


    x = np.linspace(0, 255, 100)
    y = np.exp(g(x)) / exposure_times[0]

    plt.plot(y, x, label='CRF(x)')
    plt.xlabel('x')
    plt.ylabel('CRF(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


    # hdr_image = merge_hdr(images, exposure_times, g)
    # ldr_image = tone_map(hdr_image)
    # save_image("hdr_image.JPG", ldr_image)
    
