import numpy as np
from scipy.io import loadmat, savemat
from math import gcd
import gzip
import io
import os
import fpzip

# =========================
#  Ramanujan FIR Transform
# =========================
# RFT 1D: columns are c_q(n) repeated with period q, for q=1..N
# We build F_N (N x N). Then 2D RFT is: Y = F_N^{-1} X (F_M^{-1})^T

def mobius(n: int) -> int:
    """Funzione di Möbius μ(n): 0 se n ha fattori primi ripetuti,
    (-1)^k se n è prodotto di k primi distinti, 1 se n=1."""
    if n == 1:
        return 1
    mu = 1
    p = 2
    nn = n
    while p * p <= nn:
        if nn % p == 0:
            nn //= p
            # fattore primo p trovato una volta
            mu *= -1
            if nn % p == 0:
                # fattore primo p appare due volte -> μ=0
                return 0
        p += 1 if p == 2 else 2  # piccolo speedup
    if nn > 1:
        mu *= -1
    return mu

def divisors(n: int):
    """Divisori positivi di n (non ordinati)."""
    ds = set()
    for d in range(1, int(n**0.5)+1):
        if n % d == 0:
            ds.add(d)
            ds.add(n // d)
    return ds

def ramanujan_sum_q_of_r(q: int, r: int) -> int:
    """c_q(r): formula c_q(n) = sum_{d | gcd(n,q)} d * μ(q/d). Dipende solo da gcd(n,q)."""
    g = gcd(q, r)
    s = 0
    # somma su d | g  (equivale a un cambio d = d)
    for d in divisors(g):
        mu_term = mobius(q // d)
        if mu_term != 0:
            s += d * mu_term
    return s

def build_F(N: int, dtype=np.float64) -> np.ndarray:
    """Costruisce F_N (N x N) per la RFT: colonna q (1-index) è c_q(n mod q), n=0..N-1."""
    F = np.empty((N, N), dtype=dtype)
    for q in range(1, N+1):
        # precompute c_q(r) per r=0..q-1
        cq = np.fromiter((ramanujan_sum_q_of_r(q, r) for r in range(q)), dtype=dtype, count=q)
        # colonna: ripeti cq per coprire N righe (periodo q)
        col = np.resize(cq, N)
        F[:, q-1] = col
    return F

def rft2(x: np.ndarray) -> np.ndarray:
    """RFT 2D (FIR) su matrice 2D (può essere complessa).
    Y = F_N^{-1} X (F_M^{-1})^T
    """
    assert x.ndim == 2, "Atteso array 2D"
    N, M = x.shape
    # Costruisci F_N e F_M (in double). Valori interi, ma usiamo float64 per robustezza; X può essere complessa.
    F_N = build_F(N, dtype=np.float64)
    F_M = build_F(M, dtype=np.float64)
    # Inverse (numerica)
    F_N_inv = np.linalg.inv(F_N)
    F_M_inv = np.linalg.inv(F_M)
    # Applica RFT 2D
    # Nota: x può essere complex128; numpy broadcasting gestisce bene real@complex@real^T
    Y = F_N_inv @ x @ F_M_inv.T
    return Y, (F_N, F_M, F_N_inv, F_M_inv)

def irft2(Y: np.ndarray, F_N: np.ndarray, F_M: np.ndarray) -> np.ndarray:
    """Inversa della RFT 2D: X = F_N Y F_M^T"""
    return F_N @ Y @ F_M.T

# =========================
#  Compressione
# =========================

def compress_fpzip(array: np.ndarray, out_path: str) -> bool:
    """Prova a comprimere lossless con fpzip (se disponibile). Restituisce True se riuscito."""
    try:
         # type: ignore
        # fpzip lavora su float32/float64 reali; gestiamo reale e immaginaria se complesso
        if np.iscomplexobj(array):
            real = np.ascontiguousarray(array.real)
            imag = np.ascontiguousarray(array.imag)
            cr = fpzip.compress(real)
            ci = fpzip.compress(imag)
            with open(out_path, "wb") as f:
                # header semplice: b'CFPZ' + lens + payloads
                f.write(b"CFPZ")
                f.write(np.array([real.ndim], np.uint8).tobytes())
                f.write(np.array(real.shape, np.int32).tobytes())
                f.write(np.array([len(cr), len(ci)], np.int64).tobytes())
                f.write(cr)
                f.write(ci)
        else:
            data = np.ascontiguousarray(array)
            cbytes = fpzip.compress(data)
            with open(out_path, "wb") as f:
                f.write(b"RFPZ")
                f.write(np.array([data.ndim], np.uint8).tobytes())
                f.write(np.array(data.shape, np.int32).tobytes())
                f.write(np.array([len(cbytes)], np.int64).tobytes())
                f.write(cbytes)
        return True
    except Exception:
        return False

def compress_gzip_npy(array: np.ndarray, out_path: str):
    """Fallback: salva .npy compresso con gzip (lossless)."""
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    raw = buf.getvalue()
    with gzip.open(out_path, "wb", compresslevel=9) as f:
        f.write(raw)

# =========================
#  Pipeline end-to-end
# =========================

def run_rft_compress(mat_path: str,
                     key: str = "Hol",
                     out_dir: str = "./out",
                     base_name: str = "cgh_rft"):
    """
    Carica CGH .mat (chiave 'Hol'), applica RFT 2D, salva:
      - trasformato Y (MATLAB .mat per ispezione)
      - compressione lossless (fpzip se presente, altrimenti gzip)
      - cache F_N, F_M opzionale (npy) per riuso
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Carico {mat_path} ...")
    data = loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Chiave '{key}' non trovata nel .mat")
    X = np.asarray(data[key])
    if X.ndim != 2:
        raise ValueError(f"Atteso array 2D per '{key}', trovato shape {X.shape}")
    print(f"[INFO] CGH shape: {X.shape}, dtype: {X.dtype}")

    print("[INFO] Applico RFT 2D (FIR)...")
    Y, (F_N, F_M, F_N_inv, F_M_inv) = rft2(X)

    # Salvataggi per debug/ispezione
    y_mat_path = os.path.join(out_dir, f"{base_name}_Y.mat")
    print(f"[INFO] Salvo Y per ispezione: {y_mat_path}")
    savemat(y_mat_path, {"Y": Y})

    # (opzionale) cache matrici per riuso
    np.save(os.path.join(out_dir, f"{base_name}_FN.npy"), F_N)
    np.save(os.path.join(out_dir, f"{base_name}_FM.npy"), F_M)
    # Non salviamo gli inversi (si possono ricostruire) ma puoi farlo se vuoi:
    # np.save(os.path.join(out_dir, f"{base_name}_FNinv.npy"), F_N_inv)
    # np.save(os.path.join(out_dir, f"{base_name}_FMinv.npy"), F_M_inv)

    # Compressione
    fpz_path = os.path.join(out_dir, f"{base_name}_Y.fpz")
    gz_path  = os.path.join(out_dir, f"{base_name}_Y.npy.gz")

    print("[INFO] Comprimo lossless (fpzip se disponibile)...")
    ok = compress_fpzip(Y, fpz_path)
    if ok:
        print(f"[OK] Compresso con fpzip: {fpz_path}")
    else:
        print("[WARN] fpzip non disponibile: uso gzip(.npy) come fallback.")
        compress_gzip_npy(Y, gz_path)
        print(f"[OK] Salvato gzip: {gz_path}")

    # Verifica inversione (round-trip numerico)
    print("[INFO] Test inversione (IRFT)...")
    X_rec = irft2(Y, F_N, F_M)
    err = np.max(np.abs(X_rec - X))
    print(f"[CHECK] max |X_rec - X| = {err:.3e} (dovrebbe essere ~ tolleranza floating-point)")

if __name__ == "__main__":
    # Esempio d'uso:
    file_mat = os.path.join(os.path.dirname(__file__), 'dataset', 'Hol_2D_dice.mat')
    run_rft_compress(file_mat, key="Hol", out_dir="./out", base_name="dice2d_rft")
    pass
