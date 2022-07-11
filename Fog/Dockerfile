FROM python:3.8
RUN apt-get update
RUN apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libasound-dev libsndfile1-dev -y \
        && pip3 install pyaudio
COPY requirements.txt ./requirements.txt
COPY ./templates ./templates
COPY ./templates/index.html ./templates
ADD recording1.wav .
RUN pip install -r requirements.txt
ADD trained_model_DNN.h5 .
ADD flask1.py .
ADD test.py .
CMD ["python", "./flask1.py"]
