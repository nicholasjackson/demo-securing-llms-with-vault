FROM alpine:latest

ARG TARGETARCH
ARG KAPSULE_VERSION=0.0.7

RUN apk add --no-cache curl

# Add Kapsule
RUN if [ "${TARGETARCH}" = "amd64" ]; then TARGETARCH="x86_64"; fi && \
  wget https://github.com/nicholasjackson/kapsule/releases/download/${KAPSULE_VERSION}/kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz && \
    tar -xvf kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz && \
    mv kapsule /usr/local/bin && \
    chmod +x /usr/local/bin/kapsule && \
    rm kapsule_${KAPSULE_VERSION}_linux_${TARGETARCH}.tar.gz