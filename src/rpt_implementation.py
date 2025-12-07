from scipy.io import loadmat
from scipy.io import savemat
from scipy.linalg import inv
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
from utils.utils import mobius, isPrime, divisors
from scipy.linalg import qr


def euler_phi(q): #Restituisce quanti numeri tra 1 e q sono coprimi con q
    count = 0
    for a in range(1, q + 1):
        if math.gcd(a, q) == 1:
            count += 1
    return count


def calculate_ramanujan_sums(lenght, typeRep):

    div = divisors(lenght)

    
    P_N = np.zeros((lenght, lenght), dtype=typeRep)
    
    j=0
    for q in div:
        c_q = np.zeros(q, dtype=np.float64)

        #passo 1
        for n in range(q):
            sum = 0
            g = math.gcd(q,n) 
            for d in range(1, g+1):
                if (g % d == 0):
                    sum += d*mobius(q//d) #Da ricorda che forse si mette // per la divisione senza virgola, esempio 15:2 mi da 7
            c_q[n] =  np.float64(sum) if typeRep == np.float64 else np.float32(sum)
        
        #passo 2
        coprims = euler_phi(q)

        #passo 3
        if (len(c_q) == 0):
            continue

        for l in range(coprims):
            c_q_shift = np.roll(c_q, -l)
            col = np.empty(lenght, dtype=typeRep)
            for n in range(lenght):
             #   print(c_q_shift)
                col[n] = c_q_shift[n % q]

            col_t = col.T
            
            P_N[:,j] = col_t
            j = j+1

    return P_N
            
def calculate_Y(X):

    typeRep = np.float64 if X.dtype == 'complex128' else np.float32

    P_N = calculate_ramanujan_sums(X.shape[0], typeRep)
    P_M = calculate_ramanujan_sums(X.shape[1], typeRep)


    U, S, Vt = np.linalg.svd(P_N)
    F_N_inv = np.dot(Vt.T, np.dot(np.diag(1/S), U.T))
    K, Z, Ft = np.linalg.svd(P_M)
    F_M_inv = np.dot(Ft.T, np.dot(np.diag(1/Z), K.T))

    Y = F_N_inv @ X @ F_M_inv.T 

    return Y   

def calculate_X(Y):

    typeRep = np.float64 if Y.dtype == 'complex128' else np.float32

    P_N = calculate_ramanujan_sums(Y.shape[0], typeRep)
    P_M = calculate_ramanujan_sums(Y.shape[1], typeRep)

    X = P_N @ Y @ P_M.T
    return X        
        
def compress_with_fpzip(hologram:Hologram,output_file,split=True):

    metadata = {}
    metadata["pp"]   = float(np.asarray(hologram.pp).squeeze()) #Non so perchè non li vede come scalari
    metadata["zobj"] = float(np.asarray(hologram.zobj).squeeze())
    metadata["wlen"] = float(np.asarray(hologram.wlen).squeeze())
    metadata["isSplitted"] = split
    metadata["shape"] = hologram.hol.shape

    header_bytes = json.dumps(metadata).encode("utf-8")
    header_len = np.int64(len(header_bytes)).tobytes()

    typeRep = np.float64 if hologram.hol.dtype == 'complex128' else np.float32

    with open(output_file , 'wb') as f:

        f.write(header_len)
        f.write(header_bytes)

        matrix = hologram.hol

        if split and np.iscomplexobj(matrix):
            real_data = np.ascontiguousarray(np.real(matrix), dtype=typeRep)
            imag_data = np.ascontiguousarray(np.imag(matrix), dtype=typeRep) 
            
            compressed_real = fpzip.compress(real_data, precision=0, order='C')
            compressed_imag = fpzip.compress(imag_data, precision=0, order='C')
            #Salvo la lunghezza 
            f.write(np.int64(len(compressed_real)).tobytes())
            f.write(np.int64(len(compressed_imag)).tobytes())
            f.write(compressed_real)
            f.write(compressed_imag)    
        
        else:
            #Reinterpreto la matrice complessa come una matrice di float64, senza perdere informazione
            float_view = matrix.view(typeRep)
            float_view = np.ascontiguousarray(float_view)

            compressed = fpzip.compress(float_view)
            f.write(compressed)      

def decompress_with_fpzip(output_file):
    with open(output_file, 'rb') as f:
        header_len = np.frombuffer(f.read(8), dtype=np.int64)[0]
        header_bytes = f.read(header_len)
        metadata = json.loads(header_bytes.decode("utf-8"))
        split = metadata["isSplitted"]
        shape = metadata['shape']

        if(split):
            len_real = np.frombuffer(f.read(8), dtype=np.int64)[0]
            len_imag = np.frombuffer(f.read(8), dtype=np.int64)[0]

            compressed_real = f.read(len_real)
            compressed_imaginary = f.read(len_imag)

            uncompressed_real = fpzip.decompress(compressed_real, order='C').squeeze()
            uncompressed_imaginary = fpzip.decompress(compressed_imaginary, order='C').squeeze()



            #ricostruisco la matrice complessa
            Y = uncompressed_real + 1j * uncompressed_imaginary


            return Hologram(Y, metadata["pp"], metadata["zobj"], metadata["wlen"])
        else:
            data = f.read()
            float_array = fpzip.decompress(data)
            float_array = float_array.reshape(shape[0], shape[1], 2)

            # ricostruzione dei complessi
            complex_matrix = float_array[...,0] + 1j * float_array[...,1]
            return Hologram(complex_matrix, metadata["pp"], metadata["zobj"], metadata["wlen"])





    





def calc_RPT (filename):

    ORIGINAL_CGH_FILENAME = filename





    #Recupero dell'ologramma
    filepath_mat = os.path.join(os.path.dirname(__file__),'..', 'dataset', f'{ORIGINAL_CGH_FILENAME}.mat')

    hologram_data = Hologram.open_hologram_file(filepath_mat)


    #Calcolo della trasformata dell'ologramma
    Y = calculate_Y(hologram_data.hol)
    X = hologram_data.hol


    #Si specifica che la compressione deve essere splittata, ovvero comprimere parte reale ed immaginaria in modo separato
    split = True
    output_file = os.path.join(
    os.path.dirname(__file__),
    '..',
    'out',
    f"{ORIGINAL_CGH_FILENAME}_compressed_rpt{'_unsplitted' if not split else ''}.fpzip"
)

    #Compressione utilizzando fpzip
    compress_with_fpzip(Hologram(Y,hologram_data.pp, hologram_data.zobj, hologram_data.wlen), output_file, split)

    #Decompressione utilizzando fpzip
    decompressed_hologram_data = decompress_with_fpzip(output_file)

    #Ricostruzione dell'ologramma decompresso
    decompressed_X = calculate_X(decompressed_hologram_data.hol)

    #Calcolo similarità
    similarity_manager = paper_similarity.Similarity(paper_similarity.GammaM.bump, paper_similarity.GammaR.cos,
                                             paper_similarity.GammaA.unique)

    similarity = similarity_manager.calc_similarity(X,decompressed_X)
    equality = np.array_equal(X, decompressed_X)
    difference = np.max(np.abs(X - decompressed_X))
    



    print('Similarity = ',similarity)
    print("Equality between original and decompressed hologram =", equality)
    print("Difference between original and decompressed hologram =", difference)


    #Salvo ologramma in formato .raw per calcolare il tasso di compressione
    path_raw = os.path.join(os.path.dirname(__file__),'hologram.raw')
    with open(path_raw, "wb") as fp:
        pickle.dump(hologram_data, fp)
        compression_rate = utils.calculate_compression_rate(output_file, path_raw)
    print("compression rate =", compression_rate)
    
    os.remove(path_raw)

    #show_hologram_reconstruction(hologram_data)
    #show_hologram_reconstruction(decompressed_hologram_data)

    #show_phase_and_amplitude(hologram_data)
    #show_phase_and_amplitude(decompressed_hologram_data)


    return similarity,equality,difference, compression_rate

 
