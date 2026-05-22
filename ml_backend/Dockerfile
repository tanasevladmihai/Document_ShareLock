
FROM ubuntu:22.04 AS downloader
RUN apt-get update && apt-get install -y wget
WORKDIR /models

RUN wget -O /models/gemma-4-e4b.gguf https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf

FROM ghcr.io/ggml-org/llama.cpp:server
COPY --from=downloader /models/gemma-4-e4b.gguf /models/gemma-4-e4b.gguf

CMD ["-m", "/models/gemma-4-e4b.gguf", "--host", "0.0.0.0", "--port", "8080", "--threads", "4", "--ctx-size", "2048"]
