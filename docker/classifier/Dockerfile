FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3
RUN apt-get update && apt-get -y install git
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 
RUN git clone https://github.com/jordipons/musicnn.git && cd musicnn && python setup.py install
RUN apt-get -y install libsndfile-dev
RUN pip install --no-cache-dir keras
RUN apt-get -y install sox libsox-dev
RUN pip install sox
COPY . . 
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=0
CMD flask run --host=0.0.0.0