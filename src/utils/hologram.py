import numpy as np
from scipy.io import loadmat

class Hologram:
    def __init__(self, hol:np.ndarray, pp:float, zobj:float, wlen:float):
        self.hol =  hol
        self.pp = pp 
        self.zobj = zobj
        self.wlen = wlen

    def open_hologram_file(filepath:str):

        data = loadmat(filepath)

        # self.hol = data['Hol']
        # self.pp = data.get('pitch',8.e-06)
        # self.zobj = data.get('zobj',0.9)
        # self.wlen = data.get('wlen', 6.33e-07)

        hologram = data['Hol']

        pixel_pitch = data.get('pixel_pitch') or data.get('pixelPitch') or data.get('dx') or data.get('pitch') or data.get('pp')
        if pixel_pitch is None:
            pixel_pitch = 8.e-06
            print(f"⚠ pixel_pitch non trovato, usando default: {pixel_pitch} μm")
        
        distance = data.get('distance') or data.get('z') or data.get('z0') or data.get('zobj')
        if distance is None:
            distance = 0.9
            print(f"⚠ distance non trovata, usando default: {distance} mm")

        wavelength = data.get('wavelength') or data.get('lambda') or data.get('wvl') or data.get('wlen')
        if wavelength is None:
            wavelength = 6.33e-07
            print(f"⚠ wavelength non trovata, usando default: {wavelength} nm")
        
        return Hologram(hologram, pixel_pitch, distance, wavelength)
