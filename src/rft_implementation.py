from scipy.io import loadmat
from scipy.io import savemat
import math
import numpy as np
import os
from compressors.fpzip import Fpzip
from compressors.gzip import Gzip
from compressors.bzip2 import Bzip2
from compressors.zfp import Zfp
from compressors.zip import Zip
from utils import paper_similarity
from hologram_visualization.hologram_reconstruction import *
from hologram_visualization.phase_and_amplitude_reconstruction import *
import json
from utils.hologram import Hologram
from utils.utils import calculate_compression_rate, mobius, divisors
import pickle
import time

def ramanujan_sum_for_dimension(dimension):

    ramanujan_sums= np.zeros((dimension, dimension))

    for n in range(0,dimension):
        for q in range(1,dimension+1):
            gcd = math.gcd(n,q)
            res = 0   
            for d in divisors(gcd):
                res += d*mobius(q//d)
            ramanujan_sums[n,q-1] = res
    
    return ramanujan_sums
    
def calculate_ramanujan_sums(rows_lenght, column_lenght):

    return ramanujan_sum_for_dimension(rows_lenght), ramanujan_sum_for_dimension(column_lenght)


def compress(hologram:Hologram,output_file,split,compressor):

    metadata = {}
    metadata["pp"]   = float(np.asarray(hologram.pp).squeeze()) #Non so perchè non li vede come scalari
    metadata["zobj"] = float(np.asarray(hologram.zobj).squeeze())
    metadata["wlen"] = float(np.asarray(hologram.wlen).squeeze())
    metadata["isSplitted"] = split
    metadata["original_data_type"] = str(hologram.data_type)
    metadata["shape"] = hologram.hol.shape

    header_bytes = json.dumps(metadata).encode("utf-8")
    header_len = np.int64(len(header_bytes)).tobytes()

    with open(output_file , 'wb') as f:

        f.write(header_len)
        f.write(header_bytes)

        matrix = hologram.hol

        type = np.float64
        if hologram.data_type == np.complex64:
            type = np.float32
        if split and np.iscomplexobj(matrix):
            real_data = np.ascontiguousarray(np.real(matrix),type)
            imag_data = np.ascontiguousarray(np.imag(matrix),type)

            compressed_real = compressor.compress(real_data)
            compressed_imag = compressor.compress(imag_data)

            #Salvo la lunghezza 
            f.write(np.int64(len(compressed_real)).tobytes())
            f.write(np.int64(len(compressed_imag)).tobytes())

            f.write(compressed_real)
            f.write(compressed_imag)    
        
        else:
            #Reinterpreto la matrice complessa come una matrice di float64, senza perdere informazione
            float_view = matrix.view(type)
            float_view = np.ascontiguousarray(float_view)

            compressed = compressor.compress(float_view)
            f.write(compressed)


def calculate_Y(X):

    F_N,F_M = load_ramanujan_sums(X.shape[0],X.shape[1])

    type = np.float64
    if (X.dtype == np.complex64):
        type= np.float32
    F_N_inv = np.linalg.inv(F_N).astype(type) #/!\/!\ necessario per ottenere la massima precisione possibile
    F_M_inv = np.linalg.inv(F_M).astype(type)

    X128 = np.ascontiguousarray(X.astype(np.complex128))
    Y128 = F_N_inv @ X128 @ F_M_inv.T
    Y128 = np.ascontiguousarray(Y128)  

    return Y128

def save_ramanujan_sums(F_N, F_M, F_N_output_file, F_M_output_file):

    with open(F_N_output_file, 'wb') as f:
        np.save(f,F_N)

    with open(F_M_output_file, 'wb') as f:
        np.save(f,F_M)

def load_ramanujan_sums(rows_number,columns_number):

    F_N_output_file = os.path.join(os.path.dirname(__file__),'..','ramanujan_data', f'F_N_{rows_number}.npy')
    F_M_output_file = os.path.join(os.path.dirname(__file__),'..','ramanujan_data', f'F_M_{columns_number}.npy')
    if(os.path.exists(F_N_output_file) and os.path.exists(F_M_output_file)):
        F_N = np.load(F_N_output_file)
        F_M = np.load(F_M_output_file)
    else:
        F_N,F_M = calculate_ramanujan_sums(rows_number,columns_number)
        save_ramanujan_sums(F_N,F_M,F_N_output_file,F_M_output_file)

    return F_N,F_M

def decompress(output_file,compressor):
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

            uncompressed_real = compressor.decompress(compressed_real)
            uncompressed_imaginary = compressor.decompress(compressed_imaginary)

            if(isinstance(compressor,(Gzip,Bzip2,Zip))):
                part_type = np.float64
                if metadata['original_data_type'] == 'complex64':
                    part_type = np.float32
                uncompressed_real = np.frombuffer(uncompressed_real,part_type).reshape(metadata["shape"])
                uncompressed_imaginary = np.frombuffer(uncompressed_imaginary,part_type).reshape(metadata["shape"])

            #ricostruisco la matrice complessa
            Y = uncompressed_real + 1j * uncompressed_imaginary
            return Hologram(Y, metadata["pp"], metadata["zobj"], metadata["wlen"],metadata["original_data_type"] )
        else:
            data = f.read()
            float_array = compressor.decompress(data) 
        
            if(isinstance(compressor,(Gzip,Bzip2,Zip))):
                part_type = np.float64
                if metadata['original_data_type'] == 'complex64':
                    part_type = np.float32
                float_array = np.frombuffer(float_array,part_type)

            float_array = float_array.reshape(metadata["shape"] + [2])# terza dimensione necessaria per rappresentare le due componenti del numero complesso, reale e immaginaria
            # ricostruzione dei complessi
            complex_matrix = float_array[...,0] + 1j * float_array[...,1]
            return Hologram(complex_matrix, metadata["pp"], metadata["zobj"], metadata["wlen"],metadata["original_data_type"])

def calculate_X(Y,hologram_type=np.complex128):

    F_N , F_M = load_ramanujan_sums(Y.shape[0], Y.shape[1])

    Y128 = np.ascontiguousarray(Y.astype(np.complex128))#/!\/!\ per ottenere la massima precisione possibile
    X128 = F_N @ Y128 @ F_M.T
    X128 = np.ascontiguousarray(X128)

    X64 = X128.astype(hologram_type) #/!\ ho necessità di conoscere il tipo originale dell'ologramma, per questo viene incluso nei metadata

    return X64

def calc_RFT (filename,compressor):

    #Recupero dell'ologramma
    filepath_mat = os.path.join(os.path.dirname(__file__),'..', 'dataset', f'{filename}.mat')
    hologram_data = Hologram.open_hologram_file(filepath_mat)

    #Si specifica che la compressione deve essere splittata, ovvero comprimere parte reale ed immaginaria in modo separato
    split = True
    output_file = os.path.join(
    os.path.dirname(__file__),
    '..',
    'out',
    f"{filename}_compressed_rft{'_unsplitted' if not split else ''}.{compressor.get_file_extension()}"
    )

    start_time_compress = time.perf_counter()
    #Calcolo della trasformata dell'ologramma
    Y = calculate_Y(hologram_data.hol)
    X = hologram_data.hol

    compress(Hologram(Y,hologram_data.pp, hologram_data.zobj, hologram_data.wlen,hologram_data.data_type), output_file, split,compressor)

    end_time_compress = time.perf_counter()

    start_time_decompress = time.perf_counter()
    decompressed_hologram_data = decompress(output_file,compressor)

    #Ricostruzione dell'ologramma decompresso
    decompressed_X = calculate_X(decompressed_hologram_data.hol,decompressed_hologram_data.data_type)
    end_time_decompress = time.perf_counter()
    decompressed_hologram_data.hol = decompressed_X

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
        compression_rate = calculate_compression_rate(output_file, path_raw)
    print("compression rate =", compression_rate)
    
    os.remove(path_raw)

    compress_time = end_time_compress - start_time_compress
    decompress_time = end_time_decompress - start_time_decompress

    print(f"Compression time: {compress_time} seconds")
    print(f"Decompression time: {decompress_time} seconds")
    #show_hologram_reconstruction(hologram_data)
    #show_hologram_reconstruction(decompressed_hologram_data)
    #show_phase_and_amplitude(hologram_data)
    #show_phase_and_amplitude(decompressed_hologram_data)

    return similarity,equality,difference, compression_rate,compress_time,decompress_time

    
