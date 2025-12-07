import fpzip
from compressors.compressor import Compressor

class Fpzip(Compressor):

    FILE_NAME_EXTENSION = 'fpzip'

    def compress(self, data):
        compressed_data = fpzip.compress(data)
        return compressed_data

    def decompress(self, data):
        decompressed_data = fpzip.decompress(data,order='C').squeeze()
        return decompressed_data
    
    def get_file_extension(self):
        return self.FILE_NAME_EXTENSION