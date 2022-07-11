FROM python:3.8
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD trained_model_DNN.h5 .
ADD flask1.py .
ADD test.py .
CMD ["python", "./flask1.py"]
