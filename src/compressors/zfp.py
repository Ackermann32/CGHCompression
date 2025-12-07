import zfpy
from compressors.compressor import Compressor

class Zfp(Compressor):

    FILE_NAME_EXTENSION = 'zfp'

    def compress(self, data):
        compressed_data = zfpy.compress_numpy(data)
        return compressed_data

    def decompress(self, data):
        decompressed_data = zfpy.decompress_numpy(data)
        return decompressed_data
    
    def get_file_extension(self):
        return self.FILE_NAME_EXTENSION