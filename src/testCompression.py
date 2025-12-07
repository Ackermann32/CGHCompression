import os
from rpt_implementation import calc_RPT

def CSV_generate(output_file,filename, similarity, equality, difference, algorithm, compression_rate):

    
    try:
        with open(output_file, 'a') as f:
  
            f.write(f"{filename},{algorithm},{similarity},{equality},{difference}, {compression_rate}\n")
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
  
            f.write("FILENAME, ALGORITHM, SIMILARITY, EQUALITY, DIFFERENCE, COMPRESSION RATE\n")
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")


    try:
        # Ottieni la lista dei file nella cartella dataset
        files = os.listdir(dataset_path)
        
        # Itera attraverso ogni file nella cartella
        for file_name in files:
            file_path = os.path.join(dataset_path, file_name)
            
            # Controlla se è un file (e non una sottocartella)
            if os.path.isfile(file_path):
                print("Hologram : ",file_name)
                similarity,equality,difference, compression_rate =  calc_RPT(filename=file_name.rstrip('.mat'))


                CSV_generate(output_file,file_name,similarity,equality,difference,"RPT", compression_rate)
            else:
                print(f"{file_name} non è un file (potrebbe essere una cartella).")
    
    except FileNotFoundError:
        print(f"La cartella 'dataset' non è stata trovata a {dataset_path}.")

if __name__ == '__main__':
    main()
