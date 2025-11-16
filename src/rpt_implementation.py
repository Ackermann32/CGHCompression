from scipy.io import loadmat
from scipy.io import savemat
import math
import numpy as np
import os
import fpzip
from utils import paper_similarity
import utils.utils as utils
from utils.hologram import Hologram
import json
from hologram_visualization.hologram_reconstruction import *
from hologram_visualization.phase_and_amplitude_reconstruction import *
import pickle

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

def divisors(N):
    divs = []
    for d in range(1, N + 1):
        if N % d == 0:
            divs.append(d)
    return divs

def euler_phi(q): #Restituisce quanti numeri tra 1 e q sono coprimi con q
    count = 0
    for a in range(1, q + 1):
        if math.gcd(a, q) == 1:
            count += 1
    return count


def calculate_ramanujan_sums(lenght):

    div = divisors(lenght)
    
    P_N = np.zeros((lenght, lenght))
    
    j=0
    for q in div:
        c_q = np.zeros(q)

        #passo 1
        for n in range(q):
            sum = 0
            g = math.gcd(q,n) 
            for d in range(1, g+1):
                if (g % d == 0):
                    sum += d*mobius(q//d) #Da ricorda che forse si mette // per la divisione senza virgola, esempio 15:2 mi da 7
            c_q[n] = sum
        
        #passo 2
        coprims = euler_phi(q)

        #passo 3
        if (len(c_q) == 0):
            continue

        for l in range(coprims):
            c_q_shift = np.roll(c_q, -l)
            col = np.empty(lenght)
            for n in range(lenght):
             #   print(c_q_shift)
                col[n] = c_q_shift[n % q]

            col_t = col.T
            
            P_N[:,j] = col_t
            j = j+1

    return P_N
            
def calculate_Y(X):

    P_N = calculate_ramanujan_sums(1080)
    P_M = calculate_ramanujan_sums(1920)


    F_N_inv = np.linalg.inv(P_N)
    F_M_inv = np.linalg.inv(P_M)

    Y = F_N_inv @ X @ F_M_inv.T 

    return Y   

def calculate_X(Y):
    P_N = calculate_ramanujan_sums(1080)
    P_M = calculate_ramanujan_sums(1920)

    X = P_N @ Y @ P_M.T
    return X        
        
def compress_with_fpzip(hologram:Hologram,output_file,split=True):

    metadata = {}
    metadata["pp"]   = float(np.asarray(hologram.pp).squeeze()) #Non so perchè non li vede come scalari
    metadata["zobj"] = float(np.asarray(hologram.zobj).squeeze())
    metadata["wlen"] = float(np.asarray(hologram.wlen).squeeze())
    metadata["isSplitted"] = split

    header_bytes = json.dumps(metadata).encode("utf-8")
    header_len = np.int64(len(header_bytes)).tobytes()

    with open(output_file , 'wb') as f:

        f.write(header_len)
        f.write(header_bytes)

        matrix = hologram.hol

        if split and np.iscomplexobj(matrix):
            real_data = np.ascontiguousarray(np.real(matrix), dtype=np.float64)
            imag_data = np.ascontiguousarray(np.imag(matrix), dtype=np.float64)
            
            compressed_real = fpzip.compress(real_data)
            compressed_imag = fpzip.compress(imag_data)
            #Salvo la lunghezza 
            f.write(np.int64(len(compressed_real)).tobytes())
            f.write(np.int64(len(compressed_imag)).tobytes())
            f.write(compressed_real)
            f.write(compressed_imag)    
        
        else:
            #Reinterpreto la matrice complessa come una matrice di float64, senza perdere informazione
            float_view = matrix.view(np.float64)
            float_view = np.ascontiguousarray(float_view)

            compressed = fpzip.compress(float_view)
            f.write(compressed)      

def decompress_with_fpzip(output_file):
    with open(output_file, 'rb') as f:

        header_len = np.frombuffer(f.read(8), dtype=np.int64)[0]
        header_bytes = f.read(header_len)
        metadata = json.loads(header_bytes.decode("utf-8"))
        split = metadata["isSplitted"]

        if(split):
            len_real = np.frombuffer(f.read(8), dtype=np.int64)[0]
            len_imag = np.frombuffer(f.read(8), dtype=np.int64)[0]

            compressed_real = f.read(len_real)
            compressed_imaginary = f.read(len_imag)

            uncompressed_real = fpzip.decompress(compressed_real).squeeze()
            uncompressed_imaginary = fpzip.decompress(compressed_imaginary).squeeze()

            #ricostruisco la matrice complessa
            Y = uncompressed_real + 1j * uncompressed_imaginary
            return Hologram(Y, metadata["pp"], metadata["zobj"], metadata["wlen"])
        else:
            data = f.read()
            float_array = fpzip.decompress(data)
            float_array = float_array.reshape(1080, 1920, 2)

            # ricostruzione dei complessi
            complex_matrix = float_array[...,0] + 1j * float_array[...,1]
            return Hologram(complex_matrix, metadata["pp"], metadata["zobj"], metadata["wlen"])




    





def main():
    filepath_mat = os.path.join(os.path.dirname(__file__),'..', 'dataset', f'{ORIGINAL_CGH_FILENAME}.mat')

    hologram_data = Hologram.open_hologram_file(filepath_mat)

    Y = calculate_Y(hologram_data.hol)

    X = hologram_data.hol

    split = True
    output_file = os.path.join(
    os.path.dirname(__file__),
    '..',
    'out',
    f"{ORIGINAL_CGH_FILENAME}_compressed_rpt{'_unsplitted' if not split else ''}.fpzip"
)


    compress_with_fpzip(Hologram(Y,hologram_data.pp, hologram_data.zobj, hologram_data.wlen), output_file, split)

    decompressed_hologram_data = decompress_with_fpzip(output_file)
    decompressed_hologram_data.hol = calculate_X(decompressed_hologram_data.hol)
    decompressed_X = decompressed_hologram_data.hol

    similarity_manager = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                             paper_similarity.GammaA.unique)

    similarity = similarity_manager.calc_similarity(X,decompressed_X)
    print('Similarity = ',similarity)

    #Salvo ologramma in formato .raw per calcolare il tasso di compressione
    path_raw = os.path.join(os.path.dirname(__file__),'hologram.raw')
    with open(path_raw, "wb") as fp:
        pickle.dump(hologram_data, fp)
    print("compression rate =", utils.calculate_compression_rate(output_file, path_raw))
    os.remove(path_raw)

    show_hologram_reconstruction(hologram_data)
    show_hologram_reconstruction(decompressed_hologram_data)

    show_phase_and_amplitude(hologram_data)
    show_phase_and_amplitude(decompressed_hologram_data)


 
if __name__ == '__main__':
    main()
    