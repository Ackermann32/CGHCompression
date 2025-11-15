from .HoloUtils import getComplex, hologramReconstruction
from scipy.io import loadmat
import numpy as np
import os
from utils.hologram import Hologram

def show_hologram_reconstruction(hologram:Hologram):

    #Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la visualizzazione dell'ologramma richiede una matrice quadrata
    hologram.hol = hologram.hol[:, 420:]
    hologram.hol = hologram.hol[:, :-420]

    imagMatrix = np.imag(hologram.hol)
    realMatrix = np.real(hologram.hol)

    complexMatrix = getComplex(realMatrix,imagMatrix)
    hologramReconstruction(complexMatrix, hologram.pp, hologram.zobj, hologram.wlen)

if __name__ == "__main__":
    show_hologram_reconstruction(os.path.join(os.path.dirname(__file__),'..', 'dataset', 'Hol_2D_dice.mat'))
