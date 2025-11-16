from scipy.io import loadmat
from scipy.io import savemat
import math
import numpy as np
import os
import fpzip
from utils import paper_similarity



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
    c_q = []
    P_N = np.zeros((lenght, lenght))
    
    j=0
    for q in div:

        #passo 1
        sum = 0
        for n in range(0,q-1):
            g = math.gcd(q,n) 
            for d in range(1, g+1):
                if (g % d == 0):
                    sum += d*mobius(q//d) #Da ricorda che forse si mette // per la divisione senza virgola, esempio 15:2 mi da 7
            c_q[n] = sum
        
        #passo 2
        coprims = euler_phi(q)

        #passo 3
        for l in range(0,coprims):
            c_q_shift = np.roll(c_q, -l)
            col = np.empty(lenght)
            for n in range(0, lenght-1):
                col[n] = c_q_shift[n % q]

            col_t = col.T
            
            P_N[:,j] = col_t
            j = j+1

    return P_N
            

            
        
        





    





def main():
    filepath_mat = os.path.join(os.path.dirname(__file__),'..', 'dataset', f'{ORIGINAL_CGH_FILENAME}.mat')



    P_N = calculate_ramanujan_sums(1080)

    print(P_N)

 
if __name__ == '__main__':
    main()
    