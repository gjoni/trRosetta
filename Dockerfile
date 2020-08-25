FROM python3.7

ADD . /code
RUN pip install -r /code/requirements.txt
RUN cd /code && wget https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2 && tar xf model2019_07.tar.bz2
RUN ln -s /code/predict_and_compare.py /usr/local/bin/predict_and_compare.py