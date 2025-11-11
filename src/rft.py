import numpy as np
import os
from math import gcd
from functools import lru_cache
from scipy.io import loadmat, savemat  # pip install scipy
import lzma, json


# ---------- Caricamento .MAT (adatta il nome del campo) ----------
def load_complex_hologram_from_mat(path, key=None):
    """
    Legge un .mat e restituisce H complesso (R, W), dtype complex64.
    - Se il file ha due campi 'real' e 'imag', li combina.
    - Se ha un unico array complesso, lo prende diretto.
    Se key è None, prova a indovinare.
    """
    m = loadmat(path)
    # rimuovi metadati di MATLAB
    m = {k: v for k, v in m.items() if not k.startswith("__")}
    if key is None:
        # prova a trovare il primo array 2D
        for k, v in m.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                key = k
                break
    if key is None:
        raise ValueError("Non trovo una matrice 2D nel .mat; specifica 'key'.")

    arr = m[key]
    # casi tipici: già complesso oppure reale/imag separati
    if np.iscomplexobj(arr):
        H = arr.astype(np.complex64)
    else:
        # prova campi 'real'/'imag' nello stesso dict
        real = np.imag(arr)
        imag = np.real(arr)
        if real is not None and imag is not None:
            H = real.astype(np.float32) + 1j * imag.astype(np.float32)
        else:
            raise ValueError("L'array scelto non è complesso e non trovo campi 'real'/'imag'.")
    return H

# ---------- Base Ramanujan c_q e RFT per righe ----------
def cqn(q, N):
    """Vettore c_q[n] per n=0..N-1 (periodo q)."""
    n = np.arange(N)
    ks = np.array([k for k in range(1, q+1) if gcd(k, q) == 1], dtype=np.int32)
    return np.sum(np.exp(2j*np.pi*np.outer(n, ks)/q), axis=1).astype(np.complex64)

@lru_cache(maxsize=None)
def _cqn_cached(q, N):
    return cqn(q, N)

def rft_rows_features(H, Q):
    """
    H: (R,W) complesso
    Q: iterable di periodi
    Ritorna:
      coeffs: dict q -> array complesso shape (R,) (coefficiente per riga)
      norms : dict q -> scalare ||c_q||^2
    """
    R, W = H.shape
    coeffs, norms = {}, {}
    for q in Q:
        cq = _cqn_cached(q, W)
        nrm = float(np.vdot(cq, cq).real)      # ||c_q||^2
        # prodotto scalare riga su c_q
        A_q = (H @ np.conj(cq)) / nrm          # shape (R,)
        coeffs[q] = A_q.astype(np.complex64)
        norms[q]  = nrm
    return coeffs, norms

def pack_coeffs(coeffs):
    """
    Converte {q: (R,)} -> (Q_sorted, C) con C shape (R, 2*|Q|), float32 [Re,Im,...].
    """
    Q_sorted = sorted(coeffs.keys())
    mats = []
    for q in Q_sorted:
        a = coeffs[q]  # (R,)
        mats.append(np.stack([a.real, a.imag], axis=1))  # (R,2)
    C = np.concatenate(mats, axis=1).astype(np.float32)
    return Q_sorted, C

# ---------- (Opzionale) Sintesi per ricostruire le righe ----------
def synth_rows(Q_sorted, coeffs_dict, W):
    """
    Ricostruisce le righe: Xhat[r, :] = sum_q A_q[r] * c_q.
    coeffs_dict: {q: (R,)} con le stesse R delle originali
    """
    qs = sorted(Q_sorted)
    # prendi R dalla prima voce
    first = coeffs_dict[qs[0]]
    R = first.shape[0]
    Xhat = np.zeros((R, W), dtype=np.complex64)
    for q in qs:
        cq = _cqn_cached(q, W)  # (W,)
        a = coeffs_dict[q][:, None]  # (R,1)
        Xhat += a * cq[None, :]
    return Xhat

# ---------- Salvatore "pronto a comprimere" ----------
def save_coeffs_xz(path_xz, Q_sorted, C, meta: dict):
    """
    Salva in XZ (LZMA) un pacchetto {meta, Q, C}, dove:
      - meta: dict (R, W, note, ecc.)
      - Q_sorted: lista di periodi
      - C: np.ndarray float32 (R, 2*|Q|)
    """
    # serializza in binario semplice: header json + blocco numpy
    header = {
        "meta": meta,
        "Q": Q_sorted,
        "dtype": str(C.dtype),
        "shape": C.shape
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = np.array([len(header_bytes)], dtype=np.uint32).tobytes()

    with lzma.open(path_xz, "wb", preset=6) as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(C.tobytes(order="C"))

def load_coeffs_xz(path_xz):
    with lzma.open(path_xz, "rb") as f:
        header_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        shape = tuple(header["shape"])
        dtype = np.dtype(header["dtype"])
        buf = f.read()
        C = np.frombuffer(buf, dtype=dtype).reshape(shape)
    return header, C

# ---------- ESEMPIO END-TO-END (adatta i path) ----------
if __name__ == "__main__":

    file_mat = os.path.join(os.path.dirname(__file__), 'dataset', 'Hol_2D_dice.mat')


    # 1) carica .mat
    H = load_complex_hologram_from_mat(file_mat, key='Hol')  # metti la tua chiave se serve
    R, W = H.shape
    assert (R, W) == (1080, 1920), (R, W)

    # 2) scegli i periodi Q (es. divisori "corti" di 1920, fino a 240)
    Q = [q for q in range(1, 241) if W % q == 0]

    # 3) RFT per righe
    coeffs, norms = rft_rows_features(H, Q)

    # 4) impacchetta in matrice reale (Re/Im)
    Q_sorted, C = pack_coeffs(coeffs)

    # 5) salva in formato “pronto per il codec” (qui LZMA .xz)
    meta = {"R": R, "W": W, "note": "RFT per righe, coeff complessi in (Re,Im)"}
    save_coeffs_xz("holo_RFTcoeffs.xz", Q_sorted, C, meta)

    # (OPZIONALE) ricostruzione di test dal dict 'coeffs'
    Xhat = synth_rows(Q_sorted, coeffs, W)
    # errore medio (giusto per vedere che torna la forma)
    err = np.linalg.norm(H - Xhat) / np.linalg.norm(H)
    print("Errore relativo (sintesi con i soli Q selezionati):", err)



