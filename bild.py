import numpy as np
import glob
import imageio
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
        exposure_times.append(exposure_time[0] / exposure_time[1])  # Konvertera bråktalet till float
    
    exposure_times = np.array(exposure_times, dtype=np.float32)
    return images, exposure_times

def merge_hdr(images, exposure_times):
    images = np.stack(images, axis=-1)  # Arrayen är av formen (H, W, C, N) höjd, bredd, färgkanal, bildnummer
    weights = compute_weights(images)
    log_exposures = np.log(exposure_times)
    
    # Tillämpa Debevecs HDR alogritm (antar identitetsavbildbning som CFR)
    numerator = np.sum(weights * (np.log(images + epsilon) - log_exposures.reshape(1,1,1,-1)), axis=-1)
    denominator = np.sum(weights, axis=-1)
    hdr= np.exp(numerator / (denominator + epsilon)) # exponering
    return hdr


def gsolve(Z, B, l, w):
    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros(A.shape[0])
    
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k] = wij * B[j]
            k += 1

    A[k, 128] = 1
    k += 1

    for i in range(n - 2):
        A[k, i] = l * w[i + 1]
        A[k, i + 1] = -2 * l * w[i + 1]
        A[k, i + 2] = l * w[i + 1]
        k += 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:n]
    lE = x[n:]
    return g


def compute_weights(Z):
    Zmin = np.min(Z)
    Zmax = np.max(Z)
    #weights = Z * (1 - Z)  # Polynom vikt
    #weights = np.exp(-((Z - 0.5) ** 2) / (2 * 0.1 ** 2)) # Gauss vikt
    weights = np.where(Z <= (Zmin + Zmax) / 2, Z - Zmin, Zmax - Z) # Triangulär vikt
    return weights

def compute_image_weights(images):
    image_weights  = []
    for i in range(len(images)):
        Z = images[i]
        weights = compute_weights(Z)
        image_weights.append(weights)
    return image_weights

def merge_hdr2(images, exposure_times):
    Z = np.array(images)
    Z2 = Z[:, ::100, ::100, :]
    log_exposures = np.log(exposure_times)
    w = compute_weights(np.linspace(0, 1, 256))
    g = gsolve(Z.reshape(-1, len(exposure_times)), log_exposures, 10, w)
    
    weights = 0.3
    weighted_sum = np.sum(weights * np.exp((g[Z] - log_exposures[:, np.newaxis, np.newaxis, np.newaxis])), axis=0)
    weight_sum = np.sum(weights, axis=0) + epsilon
    
    #hdr_image = np.exp(weighted_sum / weight_sum)
    return weighted_sum

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

def downsample_image(image, r):
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

def create_g_function(g_values, Zprim, delta_t):
    N = len(g_values) // len(Zprim)
    g_z = g_values[:N]

    img = Zprim[0]
    first_channel = img[:, :, 1].flatten()

    # Sortera datan för "garanterad" monoton funktion
    first_channel_sorted = np.sort(first_channel)
    g_z_sorted = np.sort(g_z)

    # Plotta funktionen
    plt.plot(first_channel_sorted, np.exp(g_z_sorted) / delta_t, label='CRF(x)')
    plt.xlabel('x')
    plt.ylabel('CRF(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Interpolera för att få en fullständig funktion
    g_interpolated = interp1d(first_channel_sorted, g_z_sorted, kind='linear', fill_value='extrapolate')

    return g_interpolated

def save_image(filename, image):
    imageio.imwrite(filename, image)

def main():
    image_files = sorted(glob.glob("bilder/*.JPG"))
    
    images, exposure_times = load_exposure_images(image_files)
    Zprim = np.array([downsample_image(image, 100) for image in images])

    #w = compute_weights(np.linspace(0, 1, 256))
    #g = gsolve(Zprim.reshape(-1, len(exposure_times)), np.log(exposure_times), 10, w)

    #create_g_function(g, Zprim,  np.log(exposure_times)[1])

    hdr_image = merge_hdr2(Zprim, exposure_times)
    ldr_image = tone_map(hdr_image)
    
    save_image("hdr_output.JPG", ldr_image)
    print("HDR bild sparad som 'hdr_output.jpg'")

if __name__ == "__main__":
    main()
