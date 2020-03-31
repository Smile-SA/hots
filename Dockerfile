FROM python:3.8-slim-buster

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get -qq update \
        && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
        python-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./requirements.txt /

COPY . /app
WORKDIR /app
RUN pip install -r /requirements.txt

RUN pip3 install --no-deps -e .


#ENTRYPOINT ["rac"]

