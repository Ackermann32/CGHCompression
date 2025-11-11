from HoloUtils import getComplex, hologramReconstruction
from scipy.io import loadmat
import numpy as np
import os

ORIGINAL_CGH_FILENAME = 'Hol_2D_dice.mat'

if __name__ == "__main__":

    filepath = os.path.join(os.path.dirname(__file__),'..', 'dataset', ORIGINAL_CGH_FILENAME)
    data = loadmat(filepath)
    print(data)

    hologram = data['Hol']
    pitch = data['pitch']
    zobj = data.get('zobj',0.9)
    wlen = data['wlen']

    #Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la visualizzazione dell'ologramma richiede una matrice quadrata
    hologram = hologram[:, 420:]
    hologram = hologram[:, :-420]

    imagMatrix = np.imag(hologram)
    realMatrix = np.real(hologram)

    complexMatrix = getComplex(realMatrix,imagMatrix)

    hologramReconstruction(complexMatrix, pitch, 0.9, wlen)