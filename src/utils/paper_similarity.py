#File importato dal progetto CPEG di Gerardo Di Pascale, Alessandro Penna, Manuel di Matteo, Francesco Paciello, Antonio Toppi
from abc import ABC
from concurrent import futures
from typing import Callable

import numpy as np

DELTA = 0.5
HEAVISIDE_PARAM = 0.5


def _relative_phase(z: complex, w: complex) -> float:
    c = np.abs(np.angle(z) - np.angle(w))
    if c <= np.pi:
        return c
    return np.abs(2 * np.pi - np.angle(z) - np.angle(w))


class GammaM(ABC):
    @staticmethod
    def bump(z, w) -> float:
        s = min(np.abs(z), np.abs(w))
        p = 3

        first_fraction = np.e / (np.e - 1)
        # +1e-10 non esiste nel paper
        exp_result = np.exp(-(1 + (s / (np.abs(np.abs(z) - np.abs(w)) + 1e-10)) ** p) ** -1)
        second_fraction = 1 / (np.e - 1)

        final_result = first_fraction * exp_result - second_fraction

        return final_result

    @staticmethod
    def inv_abs_min(z, w) -> float:
        s = min(np.abs(z), np.abs(w))
        numerator = np.abs(np.abs(z) - np.abs(w))
        return ((numerator / s) + 1) ** -1


class GammaR(ABC):
    @staticmethod
    def cos(z: complex, w: complex) -> float:
        return 0.5 * (np.cos(2 * _relative_phase(z, w)) + 1)

    @staticmethod
    def gaus_m(z: complex, w: complex) -> float:
        K = 4 / (np.pi ** 2)
        d = 10 ** -4
        a = K * np.log((1 + d) / d)
        return (1 + d) * np.exp(-a * _relative_phase(z, w) ** 2) - d

    @staticmethod
    def abs_cos(z: complex, w: complex) -> float:
        return np.abs(np.cos(_relative_phase(z, w)))


class GammaA(ABC):
    @staticmethod
    def unique(z: complex, w: complex) -> float:
        h = np.heaviside(np.pi / 2 - _relative_phase(z, w), HEAVISIDE_PARAM)
        return DELTA + (1 - DELTA) * h


class Similarity:
    def __init__(self, gamma_M: Callable[[complex, complex], float], gamma_r: Callable[[complex, complex], float],
                 gamma_a: Callable[[complex, complex], float]):
        self.gamma_M = gamma_M
        self.gamma_r = gamma_r
        self.gamma_a = gamma_a

    def similarity_score(self, z: complex, w: complex):
        return self.gamma_M(z, w) * self.gamma_r(z, w) * self.gamma_a(z, w)

    def calc_similarity(self, matrix1: np.matrix, matrix2: np.matrix):
        num_processes = multiprocessing.cpu_count()
        chunks1 = np.array_split(matrix1, num_processes)
        chunks2 = np.array_split(matrix2, num_processes)

        with futures.ProcessPoolExecutor() as executor:
            fut = [executor.submit(np.vectorize(self.similarity_score), c1, c2) for c1, c2 in zip(chunks1, chunks2)]
            results = []
            for future in futures.as_completed(fut):
                result_chunk = future.result()
                results.append(result_chunk)
        result_matrix = np.concatenate(results, axis=0)
        return np.mean(result_matrix)
