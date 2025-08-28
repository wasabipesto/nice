# Dockerfile for building on Debian Trixie (GLIBC 2.41)
FROM rust:trixie
RUN apt-get update
RUN apt-get install -y build-essential

WORKDIR /opt/nice
ENV CARGO_TARGET_DIR=/opt/nice/target-trixie
ENTRYPOINT ["cargo", "build", "-r"]
