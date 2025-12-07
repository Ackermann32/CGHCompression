import bz2
from compressors.compressor import Compressor

class Bzip2(Compressor):

    FILE_NAME_EXTENSION = 'bz2'

    def compress(self, data):
        compressed_data = bz2.compress(data,compresslevel=9)
        return compressed_data

    def decompress(self, data):
        decompressed_data = bz2.decompress(data)
        return decompressed_data
    
    def get_file_extension(self):
        return self.FILE_NAME_EXTENSION