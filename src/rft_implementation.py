from scipy.io import loadmat
import math
import numpy as np
import os
import fpzip

ORIGINAL_CGH_FILENAME = 'Hol_2D_dice.mat'

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
    



def ramanujan_sums(rows_lenght, column_lenght):

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

def compress_with_fpzip(matrix,output_file,split):

    if split and np.iscomplexobj(matrix):
        real_data = np.ascontiguousarray(np.real(matrix), dtype=np.float64)
        imag_data = np.ascontiguousarray(np.imag(matrix), dtype=np.float64)
        with open(output_file , 'wb') as f:
            compressed_real = fpzip.compress(real_data)
            compressed_imag = fpzip.compress(imag_data)
            f.write(compressed_real)
            f.write(compressed_imag)
    
    ''' TODO implementare eventualmente la compressione senza split di parte reale e immaginaria per confrontare
    else:
        data = np.ascontiguousarray(matrix, dtype=np.float64)
        with open(output_file, 'wb') as f:
            compressed = fpzip.compress(data)
            f.write(compressed)
    '''
def main():
    file_mat = os.path.join(os.path.dirname(__file__), 'dataset', ORIGINAL_CGH_FILENAME)
    data = loadmat(file_mat)
    X = data["Hol"]

    F_N, F_M = ramanujan_sums(1080, 1920)

    F_N_inv = np.linalg.inv(F_N)
    F_M_inv = np.linalg.inv(F_M)

    Y = F_N_inv @ X @ F_M_inv.T     

    output_file = os.path.join(os.path.dirname(__file__), 'out', f'{ORIGINAL_CGH_FILENAME}_compressed.fpzip')
    
    compress_with_fpzip(Y, output_file, split=True)

if __name__ == '__main__':
    main()
    
