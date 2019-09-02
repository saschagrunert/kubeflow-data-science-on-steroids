FROM python:latest
COPY test/images test/images

RUN pip install --no-cache-dir \
    cmake \
    torchvision

RUN pip install --no-cache-dir \
    dill==0.3.0 \
    pandas==0.24.2 \
    matplotlib==3.1.1 \
    scipy==1.3.0 \
    numpy==1.16.4 \
    plotly==4.0.0 \
    recordclass==0.11.1 \
    fastai==1.0.57
