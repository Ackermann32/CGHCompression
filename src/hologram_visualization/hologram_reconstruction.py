from .HoloUtils import getComplex, hologramReconstruction
from scipy.io import loadmat
import numpy as np
import os

def show_hologram_reconstruction(filepath):

    data = loadmat(filepath)

    hologram = data['Hol']
    pitch = data.get('pitch',8.e-06)
    zobj = data.get('zobj',0.9)
    wlen = data.get('wlen', 6.33e-07)

    #Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la visualizzazione dell'ologramma richiede una matrice quadrata
    hologram = hologram[:, 420:]
    hologram = hologram[:, :-420]

    imagMatrix = np.imag(hologram)
    realMatrix = np.real(hologram)

    complexMatrix = getComplex(realMatrix,imagMatrix)
    hologramReconstruction(complexMatrix, pitch, zobj, wlen)

if __name__ == "__main__":
    show_hologram_reconstruction(os.path.join(os.path.dirname(__file__),'..', 'dataset', 'Hol_2D_dice.mat'))
