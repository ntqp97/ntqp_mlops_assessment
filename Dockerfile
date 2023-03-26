# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /
ADD . .
# Install git
RUN apt-get update && apt-get install -y git
# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD python3 -u app.py
