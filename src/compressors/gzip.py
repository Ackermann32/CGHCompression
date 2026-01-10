import gzip
from compressors.compressor import Compressor

class Gzip(Compressor):

    FILE_NAME_EXTENSION = 'gz'

    def compress(self, data):
        compressed_data = gzip.compress(data,compresslevel=9)
        return compressed_data

    def decompress(self, data):
        decompressed_data = gzip.decompress(data)
        return decompressed_data
    
    def get_file_extension(self):
        return self.FILE_NAME_EXTENSION