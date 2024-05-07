# Base with installed python dependencies
FROM python:3.11-slim AS python-base

# Install & use pipenv
RUN python -m pip install --upgrade pip
RUN pip install pipenv

# Tell pipenv to create venv in the current directory
ENV PIPENV_VENV_IN_PROJECT=1

# Pipfile contains requests
ADD Pipfile.lock Pipfile /usr/src/

WORKDIR /usr/src

RUN pipenv sync

# Install and cache the base model from huggingface
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

ADD install_model.py /usr/src
RUN ./.venv/bin/python install_model.py

# Add llama CPP
FROM ghcr.io/ggerganov/llama.cpp:full AS llama-cpp

# Runtime image
FROM docker.io/python:3.11 AS runtime

ARG TARGETARCH
ARG KAPSULE_VERSION=0.0.5

# Add llama CPP
COPY --from=llama-cpp /app /llama_cpp

# Add Kapsule
RUN if [ "${TARGETARCH}" = "amd64" ]; then TARGETARCH="x86_64"; fi && \
  wget https://github.com/nicholasjackson/kapsule/releases/download/${KAPSULE_VERSION}/kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz && \
    tar -xvf kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz && \
    mv kapsule /usr/local/bin && \
    chmod +x /usr/local/bin/kapsule && \
    rm kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz

# Add the python virtual env
RUN mkdir -v /usr/src/.venv
COPY --from=python-base /usr/src/.venv/ /usr/src/.venv/

WORKDIR /usr/src

# To use venv run with ./.venv/bin/python file.py