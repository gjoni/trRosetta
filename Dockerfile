FROM python:3.7

RUN apt-get update && apt-get install -yf dssp\
 && rm -rf /var/lib/apt/lists/*

ADD . /code
RUN pip install -r /code/requirements.txt
RUN cd /code && wget https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2 && tar xf model2019_07.tar.bz2
RUN ln -s /code/predict_and_compare.py /usr/local/bin/predict_and_compare.py
