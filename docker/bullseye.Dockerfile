# Dockerfile for building on Debian Bullseye (GLIBC 2.31)
FROM rust:bullseye
RUN apt-get update
RUN apt-get install -y build-essential

WORKDIR /opt/nice
ENV CARGO_TARGET_DIR=/opt/nice/target-bullseye
ENTRYPOINT ["cargo", "build", "-r"]
