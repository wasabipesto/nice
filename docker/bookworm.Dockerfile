# Dockerfile for building on Debian Bookworm (GLIBC 2.36)
FROM rust:bookworm
RUN apt-get update
RUN apt-get install -y build-essential

COPY . /opt/nice
WORKDIR /opt/nice

ENV CARGO_TARGET_DIR=/opt/nice/target-bookworm
ENTRYPOINT ["cargo", "build", "-r"]
