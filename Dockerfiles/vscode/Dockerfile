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

# Add llama CPP
FROM ghcr.io/ggerganov/llama.cpp:full AS llama-cpp

FROM ghcr.io/jumppad-labs/vscode:base-v0.4.0

ARG TARGETARCH
ARG KAPSULE_VERSION=0.0.5

# Install default packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
  curl \
  wget \
  zip \
  git \
  jq \
  build-essential \
  postgresql-client \
  python3-full \
  python3-pip \
  pipx \
  openssh-server

# Clean packages
RUN rm -rf /var/lib/apt/lists/*

# Add llama CPP
COPY --from=llama-cpp /app /llama_cpp

# Install Vault
RUN wget -O vault.zip https://releases.hashicorp.com/vault/1.16.2/vault_1.16.2_linux_${TARGETARCH}.zip && \
  unzip vault.zip && \
  mv vault /usr/local/bin

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/${TARGETARCH}/kubectl" && \
  mv ./kubectl /usr/local/bin/kubectl && \
  chmod +x /usr/local/bin/kubectl

# Add Kapsule
RUN if [ "${TARGETARCH}" = "amd64" ]; then TARGETARCH="x86_64"; fi && \
  wget https://github.com/nicholasjackson/kapsule/releases/download/${KAPSULE_VERSION}/kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz && \
    tar -xvf kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz && \
    mv kapsule /usr/local/bin && \
    chmod +x /usr/local/bin/kapsule && \
    rm kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz

# Install VSCode extensions
RUN code-server --install-extension github.github-vscode-theme && \
    code-server --install-extension pkief.material-icon-theme && \
    code-server --install-extension ms-python.python && \
    code-server --install-extension ms-python.debugpy && \
    code-server --install-extension ms-toolsai.jupyter

# Add the python virtual env
RUN mkdir -v /usr/src/.venv
COPY --from=python-base /usr/src/.venv/ /usr/src/.venv/

WORKDIR /usr/src