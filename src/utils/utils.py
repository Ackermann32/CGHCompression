import os

def calculate_compression_rate(compressed_filepath: str, uncompressed_filepath: str) -> float:
    compressed_len = os.path.getsize(compressed_filepath)
    uncompressed_len = os.path.getsize(uncompressed_filepath)
    ratio = compressed_len / uncompressed_len
    return ratio