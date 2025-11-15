import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os
from utils.hologram import Hologram

def reconstruct_with_angular_spectrum(hologram, pixel_pitch, wavelength, distance):
    """
    Ricostruzione usando il metodo Angular Spectrum
    Questo è il metodo più accurato per ologrammi digitali
    """
    M, N = hologram.shape
    
    # Converti in campo complesso (se l'ologramma è reale)
    if not np.iscomplexobj(hologram):
        hologram = hologram.astype(complex)
    
    # Griglia di frequenze spaziali
    fx = np.fft.fftfreq(N, pixel_pitch)
    fy = np.fft.fftfreq(M, pixel_pitch)
    FX, FY = np.meshgrid(fx, fy)
    
    # Trasformata di Fourier dell'ologramma
    H = fft2(hologram)
    
    # Funzione di trasferimento dell'Angular Spectrum
    # H(fx, fy) = exp(i * k * z * sqrt(1 - (λ*fx)² - (λ*fy)²))
    k = 2 * np.pi / wavelength
    
    # Calcola la radice quadrata con protezione per valori negativi
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    
    # Evanescent waves filter (frequenze troppo alte vengono filtrate)
    mask = arg >= 0
    
    sqrt_arg = np.zeros_like(arg, dtype=complex)
    sqrt_arg[mask] = np.sqrt(arg[mask])
    sqrt_arg[~mask] = 1j * np.sqrt(-arg[~mask])  # Onde evanescenti
    
    # Funzione di trasferimento
    transfer_function = np.exp(1j * k * distance * sqrt_arg)
    
    # Applica la propagazione
    U = H * transfer_function
    
    # Trasformata inversa per ottenere il campo ricostruito
    reconstructed_field = ifft2(U)
    
    return reconstructed_field

def display_reconstruction(hologram, reconstructed_field, metadata):
    """Visualizza i risultati della ricostruzione"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Ologramma originale
    axes[0, 0].imshow(np.abs(hologram), cmap='gray')
    axes[0, 0].set_title('Ologramma Originale')
    axes[0, 0].axis('off')
    
    # Spettro di Fourier
    H = fftshift(fft2(hologram))
    axes[0, 1].imshow(np.log1p(np.abs(H)), cmap='viridis')
    axes[0, 1].set_title('Spettro di Fourier (log)')
    axes[0, 1].axis('off')
    
    # Fase dell'ologramma
    axes[0, 2].imshow(np.angle(hologram), cmap='twilight')
    axes[0, 2].set_title('Fase Ologramma')
    axes[0, 2].axis('off')
    
    # Ampiezza ricostruita
    amplitude = np.abs(reconstructed_field)
    axes[1, 0].imshow(amplitude, cmap='gray')
    axes[1, 0].set_title('Ampiezza Ricostruita')
    axes[1, 0].axis('off')
    
    # Intensità ricostruita
    intensity = amplitude**2
    axes[1, 1].imshow(intensity, cmap='gray')
    axes[1, 1].set_title('Intensità Ricostruita')
    axes[1, 1].axis('off')
    
    # Fase ricostruita
    axes[1, 2].imshow(np.angle(reconstructed_field), cmap='twilight')
    axes[1, 2].set_title('Fase Ricostruita')
    axes[1, 2].axis('off')
    
    # Aggiungi info metadata
    info_text = "Parametri:\n"
    for key, value in metadata.items():
        if 'pitch' in key.lower():
            info_text += f"{key}: {value} μm\n"
        elif 'wavelength' in key.lower() or 'lambda' in key.lower():
            info_text += f"{key}: {value} nm\n"
        elif 'distance' in key.lower() or key in ['z', 'z0']:
            info_text += f"{key}: {value} mm\n"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def show_phase_and_amplitude(hologram:Hologram): 

    
    try:    
        # Richiedi parametri mancanti      
        metadata_used = {
            'pixel_pitch': hologram.pp,
            'wavelength': hologram.wlen,
            'distance': hologram.zobj
        }
        
        print("\nRicostruzione in corso...")
        
        # Ricostruisci con Angular Spectrum
        reconstructed = reconstruct_with_angular_spectrum(
            hologram.hol, hologram.pp, hologram.wlen, hologram.zobj
        )
        
        # Visualizza risultati
        display_reconstruction(hologram.hol, reconstructed, metadata_used)

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

# Esempio di utilizzo
if __name__ == "__main__":
    show_phase_and_amplitude(os.path.join(os.path.dirname(__file__),'..', 'dataset', 'Hol_2D_dice.mat'))
