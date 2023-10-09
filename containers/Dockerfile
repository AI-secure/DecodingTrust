FROM huggingface/transformers-pytorch-gpu:latest
LABEL authors="Chenhui Zhang"
LABEL description="Docker image for running DecodingTrust with the dt-run command"

RUN git clone https://github.com/AI-secure/DecodingTrust.git
WORKDIR DecodingTrust
RUN git checkout release && pip install -e ".[quant]" && mkdir ./.cache

ENTRYPOINT ["dt-run"]
