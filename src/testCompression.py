import os
from compressors.fpzip import Fpzip
from compressors.gzip import Gzip
from compressors.bzip2 import Bzip2
from compressors.zfp import Zfp
from compressors.zip import Zip
from rft_implementation import calc_RFT
from rpt_implementation import calc_RPT

def CSV_generate(output_file,filename, similarity, equality, difference, algorithm, compression_rate, compressor,compression_time,decompression_time):

    
    try:
        with open(output_file, 'a') as f:
  
            f.write(f"{filename},{algorithm},{similarity},{equality},{difference},{compression_rate},{compressor},{compression_time},{decompression_time}\n")
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")

def main():
    # Costruisci il percorso della cartella dataset, che è due livelli sopra
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')

    output_file = os.path.join(
        os.path.dirname(__file__),
        '..',
        'out',
        "report.csv"
    )

    try:
        if output_file:
            os.remove(output_file)
    except:
        print('Il file non esiste')

    try:
        with open(output_file, 'a') as f:
  
            f.write("FILENAME, ALGORITHM, SIMILARITY, EQUALITY, DIFFERENCE, COMPRESSION RATE, COMPRESSOR, COMPRESSION TIME, DECOMPRESSION TIME\n")
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")


    try:
        # Ottieni la lista dei file nella cartella dataset
        files = os.listdir(dataset_path)
        
        compressors = (Fpzip(), Gzip(), Bzip2(), Zfp(), Zip())

        # Itera attraverso ogni file nella cartella
        for file_name in files:
            file_path = os.path.join(dataset_path, file_name)
            
            # Controlla se è un file (e non una sottocartella)
            if os.path.isfile(file_path):
                print("Hologram : ",file_name)
                for compressor in compressors:
                    similarity,equality,difference, compression_rate,compression_time,decompression_time =  calc_RPT(filename=file_name.rstrip('.mat'), compressor=compressor)
                    CSV_generate(output_file,file_name,similarity,equality,difference,"RPT", compression_rate,type(compressor).__name__,compression_time,decompression_time)
                    similarity,equality,difference, compression_rate,compression_time,decompression_time =  calc_RFT(filename=file_name.rstrip('.mat'), compressor=compressor)
                    CSV_generate(output_file,file_name,similarity,equality,difference,"RFT", compression_rate,type(compressor).__name__,compression_time,decompression_time)

            else:
                print(f"{file_name} non è un file (potrebbe essere una cartella).")
    
    except FileNotFoundError:
        print(f"La cartella 'dataset' non è stata trovata a {dataset_path}.")

if __name__ == '__main__':
    main()
