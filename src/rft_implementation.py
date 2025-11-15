from scipy.io import loadmat
from scipy.io import savemat
import math
import numpy as np
import os
import fpzip
from utils import paper_similarity
from hologram_visualization.hologram_reconstruction import *
from hologram_visualization.phase_and_amplitude_reconstruction import *

ORIGINAL_CGH_FILENAME = 'Hol_2D_dice'

def isPrime(n) :

    if (n < 2) :
        return False
    for i in range(2, n + 1) :
        if (i * i <= n and n % i == 0) :
            return False
    return True

def mobius(N) :
    
    # Base Case
    if (N == 1) :
        return 1

    # For a prime factor i 
    # check if i^2 is also
    # a factor.
    p = 0
    for i in range(1, N + 1) :
        if (N % i == 0 and 
                isPrime(i)) :

            # Check if N is
            # divisible by i^2
            if (N % (i * i) == 0) :
                return 0
            else :

                # i occurs only once, 
                # increase p
                p = p + 1

    # All prime factors are
    # contained only once
    # Return 1 if p is even
    # else -1
    if(p % 2 != 0) :
        return -1
    else :
        return 1
    
def ramanujan_sum_for_dimension(dimension):
    res = 0
    for n in range(0,dimension):
        for q in range(1,dimension+1):
            gcd = math.gcd(n,q)
            for d in range(1, gcd+1):
                if (gcd % d == 0):
                    res += d*mobius(q/d)
    
    return res
    


#TODO ottimizzare 
def calculate_ramanujan_sums(rows_lenght, column_lenght):

    ramanujan_sums_row = np.zeros((rows_lenght, rows_lenght))
    ramanujan_sums_column = np.zeros((column_lenght,column_lenght))

    for n in range (0, rows_lenght):
        for q in range(1,rows_lenght+1):
            res = 0
            gcd = math.gcd(n,q)
            for d in range(1, gcd+1):
                if (gcd % d == 0):
                    res += d*mobius(q//d) #Da ricorda che forse si mette // per la divisione senza virgola, esempio 15:2 mi da 7

            ramanujan_sums_row[n,q-1] = res

    for n in range (0, column_lenght):
        for q in range(1,column_lenght+1):
            res = 0
            gcd = math.gcd(n,q)
            for d in range(1, gcd+1):
                if (gcd % d == 0):
                    res += d*mobius(q//d)

            ramanujan_sums_column[n,q-1] = res
    
    return ramanujan_sums_row, ramanujan_sums_column

def compress_with_fpzip(matrix,output_file,split=True):

    if split and np.iscomplexobj(matrix):
        real_data = np.ascontiguousarray(np.real(matrix), dtype=np.float64)
        imag_data = np.ascontiguousarray(np.imag(matrix), dtype=np.float64)
        with open(output_file , 'wb') as f:
            compressed_real = fpzip.compress(real_data)
            compressed_imag = fpzip.compress(imag_data)
            #Salvo la lunghezza 
            f.write(np.array([len(compressed_real)], dtype=np.int64).tobytes())
            f.write(np.array([len(compressed_imag)], dtype=np.int64).tobytes())
            f.write(compressed_real)
            f.write(compressed_imag)    
    
    else:
        #Reinterpreto la matrice complessa come una matrice di float64, senza perdere informazione
        float_view = matrix.view(np.float64)
        float_view = np.ascontiguousarray(float_view)

        with open(output_file, 'wb') as f:
            compressed = fpzip.compress(float_view)
            f.write(compressed)


def calculate_Y(X):
    F_N_output_file = os.path.join(os.path.dirname(__file__),'..','ramanujan_data', 'F_N.npy')
    F_M_output_file = os.path.join(os.path.dirname(__file__),'..','ramanujan_data', 'F_M.npy')
    if(os.path.exists(F_N_output_file) and os.path.exists(F_M_output_file)):
        F_N,F_M = load_ramanujan_sums()
    else:   
        F_N,F_M = calculate_ramanujan_sums(1080,1920)
        save_ramanujan_sums(F_N,F_M)

    F_N_inv = np.linalg.inv(F_N)
    F_M_inv = np.linalg.inv(F_M)

    Y = F_N_inv @ X @ F_M_inv.T 

    return Y   

def save_ramanujan_sums(F_N, F_M, F_N_output_file, F_M_output_file):

    with open(F_N_output_file, 'wb') as f:
        np.save(f,F_N)

    with open(F_M_output_file, 'wb') as f:
        np.save(f,F_M)

def load_ramanujan_sums():
    F_N_output_file = os.path.join(os.path.dirname(__file__),'..','ramanujan_data', 'F_N.npy')
    F_M_output_file = os.path.join(os.path.dirname(__file__),'..','ramanujan_data', 'F_M.npy')
    F_N = np.load(F_N_output_file)
    F_M = np.load(F_M_output_file)

    return F_N,F_M

def decompress_with_fpzip(output_file,split=True):
    with open(output_file, 'rb') as f:

        if(split):
            len_real = np.frombuffer(f.read(8), dtype=np.int64)[0]
            len_imag = np.frombuffer(f.read(8), dtype=np.int64)[0]

            compressed_real = f.read(len_real)
            compressed_imaginary = f.read(len_imag)

            uncompressed_real = fpzip.decompress(compressed_real).squeeze()
            uncompressed_imaginary = fpzip.decompress(compressed_imaginary).squeeze()

            #ricostruisco la matrice complessa
            Y = uncompressed_real + 1j * uncompressed_imaginary
            return Y
        else:
            data = f.read()
            float_array = fpzip.decompress(data)
            float_array = float_array.reshape(1080, 1920, 2)

            # ricostruzione dei complessi
            complex_matrix = float_array[...,0] + 1j * float_array[...,1]
            return complex_matrix

def calculate_X(Y):
    F_N , F_M = load_ramanujan_sums()
    X = F_N @ Y @ F_M.T
    return X

def main():
    filepath_mat = os.path.join(os.path.dirname(__file__),'..', 'dataset', f'{ORIGINAL_CGH_FILENAME}.mat')
    data = loadmat(filepath_mat)
    X = data["Hol"]

    Y = calculate_Y(X)
    split = True
    output_file = os.path.join(os.path.dirname(__file__),'..', 'out', f'{ORIGINAL_CGH_FILENAME}_compressed{'_unsplitted' if split == False else ''}.fpzip')  
    compress_with_fpzip(Y, output_file, split)

    # Risulta essere più efficiente la compressione separando parte reale e immaginaria
    # split = False
    # output_file = os.path.join(os.path.dirname(__file__), 'out', f'{ORIGINAL_CGH_FILENAME}_compressed{'_unsplitted' if split == False else ''}.fpzip')  
    # compress_with_fpzip(Y, output_file, split)

    # print('Size splitted:',os.path.getsize(os.path.join(os.path.dirname(__file__), 'out', f'{ORIGINAL_CGH_FILENAME}_compressed.fpzip')  ))
    # print('Size unsplitted:',os.path.getsize(os.path.join(os.path.dirname(__file__), 'out', f'{ORIGINAL_CGH_FILENAME}_compressed_unsplitted.fpzip')  ))

    Y = decompress_with_fpzip(output_file,split)

    decompressed_X = calculate_X(Y)
    decompressed_filepath_mat = os.path.join(os.path.dirname(__file__),'..', 'decompressed', f'{ORIGINAL_CGH_FILENAME}_decompressed.mat')
    savemat(decompressed_filepath_mat, {"Hol": decompressed_X})

    similarity_manager = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                             paper_similarity.GammaA.unique)

    similarity = similarity_manager.calc_similarity(X,decompressed_X)
    print('Similarity = ',similarity)

    show_hologram_reconstruction(filepath_mat)
    show_hologram_reconstruction(decompressed_filepath_mat)

    show_phase_and_amplitude(filepath_mat)
    show_phase_and_amplitude(decompressed_filepath_mat)

if __name__ == '__main__':
    main()
    
