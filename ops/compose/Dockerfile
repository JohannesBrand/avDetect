FROM python:3.7

WORKDIR /avdetect_build

COPY requirements.txt /avdetect_build/

# Get python dependencies  
RUN \
  pip install --upgrade pip && \
  pip install -r requirements.txt 

COPY src /avdetect_build/src

RUN [ -n "$(find . -name __pycache__)" ] && find . -name __pycache__ | xargs rm -r || true

CMD python /avdetect_build/src/runModel.py --channel=$ENDPOINT --storage-service-channel=$STORAGE_SERVICE_ENDPOINT
EXPOSE 50061/tcp

