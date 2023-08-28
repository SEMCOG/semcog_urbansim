FROM python/3.9.18-alpine3.18
LABEL maintainer="SEMCOG"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg build-essential
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN git clone https://github.com/SEMCOG/semcog_urbansim && cd semcog_urbansim && git checkout forecast_2050

RUN git clone https://github.com/SEMCOG/urbansim.git && cd urbansim && git checkout dev
RUN python3 -m pip install --no-cache-dir -e ./urbansim

RUN git clone https://github.com/SEMCOG/urbansim_parcels.git && cd urbansim_parcels && git checkout master
RUN python3 -m pip install --no-cache-dir -e ./urbansim_parcels

RUN git clone https://github.com/UDST/urbansim_templates.git && cd urbansim_templates && git checkout dev
RUN python3 -m pip install --no-cache-dir -e ./urbansim_templates

RUN git clone https://github.com/UDST/pandana.git && cd pandana && git checkout dev
RUN python3 -m pip install --no-cache-dir -e ./pandana

RUN git clone https://github.com/UDST/developer && cd developer && git checkout master
RUN python3 -m pip install --no-cache-dir -e ./developer

RUN git clone https://github.com/UDST/choicemodels.git && cd choicemodels && git checkout dev
RUN python3 -m pip install --no-cache-dir -e ./choicemodels

RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python3 -m pip install tqdm pandas==1.5.3 carto==1.11.3 cartoframes==1.2.4 sklearn==0.0 openpyxl wheel