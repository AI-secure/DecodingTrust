FROM ubuntu:latest
LABEL authors="danielz"

ENTRYPOINT ["top", "-b"]