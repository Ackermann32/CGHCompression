# CGHCompression

# Init Environment:

## 1 packet install venv/pip per 3.11  
sudo apt update  
sudo apt install -y python3.11 python3.11-venv python3.11-distutils  

## 2 make environment
python3.11 -m venv .venv  

## 3 Activate and install all dipendences
source .venv/bin/activate  
pip install -r requirements.txt  

# Run tests:
cd src  
python3 ./testCompression.py  

# Folders  
/dataset folder contains the holograms to be compressed  
/out folder contains the compressed files and the generated report.csv with the results  