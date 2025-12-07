import zlib #valutiamo l'algoritmo di compressione DEFLATE,ma non siamo intenzionati a creare un archivio, quindi usiamo zlib invece di zipfile
from compressors.compressor import Compressor

class Zip(Compressor):

    FILE_NAME_EXTENSION = 'zip'

    def compress(self, data):
        compressed_data = zlib.compress(data)
        return compressed_data

    def decompress(self, data):
        decompressed_data = zlib.decompress(data)
        return decompressed_data
    
    def get_file_extension(self):
        return self.FILE_NAME_EXTENSION