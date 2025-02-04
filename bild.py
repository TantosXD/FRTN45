import numpy as np
import glob
import imageio
import piexif

epsilon = 1e-6

def load_exposure_images(image_files):
    images = []
    exposure_times = []
    
    for img_path in image_files:
        image = imageio.v2.imread(img_path).astype(np.float32) / 255.0
        images.append(image)
        
        exif_data = piexif.load(img_path)
        exposure_time = exif_data['Exif'][piexif.ExifIFD.ExposureTime]
        exposure_times.append(exposure_time[0] / exposure_time[1])  # Konvertera bråktalet till float
    
    exposure_times = np.array(exposure_times, dtype=np.float32)
    return images, exposure_times

def compute_weights(Z):
    #weights = Z * (1 - Z)  # Polynom vikt
    #weights = np.exp(-((Z - 0.5) ** 2) / (2 * 0.2 ** 2)) # Gauss vikt
    weights = np.where(Z <= 0.5, Z, 1 - Z) # Triangulär vikt
    return weights

def merge_hdr(images, exposure_times):
    images = np.stack(images, axis=-1)  # Arrayen är av formen (H, W, C, N) höjd, bredd, färgkanal, bildnummer
    weights = compute_weights(images)
    log_exposures = np.log(exposure_times)
    
    # Tillämpa Debevecs HDR alogritm
    numerator = np.sum(weights * (np.log(images + epsilon) - log_exposures.reshape(1,1,1,-1)), axis=-1)
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

def main():
    image_files = sorted(glob.glob("bilder/*.JPG"))
    
    images, exposure_times = load_exposure_images(image_files)
    hdr_image = merge_hdr(images, exposure_times)
    ldr_image = tone_map(hdr_image)
    
    save_image("hdr_output.JPG", ldr_image)
    print("HDR bild sparad som 'hdr_output.jpg'")

if __name__ == "__main__":
    main()
