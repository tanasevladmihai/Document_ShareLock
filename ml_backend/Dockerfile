FROM ghcr.io/ggml-org/llama.cpp:server

USER root
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

ENTRYPOINT []

CMD ["/bin/bash", "-c", " \
    mkdir -p /models && \
    if [ ! -f /models/gemma-4-e4b.gguf ]; then \
        echo 'Downloading Gemma 4B model inside the container...'; \
        wget -O /models/gemma-4-e4b.gguf https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf; \
    fi && \
    echo 'Starting Llama Server...'; \
    exec /app/llama-server -m /models/gemma-4-e4b.gguf --host 0.0.0.0 --port 8080 --threads 4 --ctx-size 1024 --no-mmap"]
