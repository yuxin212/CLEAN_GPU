FROM python:3.10

WORKDIR /app

COPY app /app

RUN pip install --upgrade -r requirements.txt

# Ubuntu installation of Pytorch
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir data/esm_data

RUN mkdir data/pretrained

# Download weights in data/pretrained
RUN pip install gdown
RUN gdown --id 1zrEU-HPNV3wp7wLAx4KnuiVyD794Oboj
RUN apt-get update && apt-get install -y unzip
RUN unzip CLEAN_pretrained.zip -d data/pretrained

RUN python build.py install

# Initialize the large weights
# RUN python CLEAN_infer_fasta.py --fasta_data price

