# Document_ShareLock

Document_ShareLock is a **highly private**, containerized **LLM application** for asking questions about uploaded documents and getting grounded answers from a local model. It supports *TXT*, *PDF*, and *DOCX* uploads, token-based chunking, local embeddings, retrieval, whole-document summaries, and two answer styles: *Precise* and *Insightful*.

## Overview

- Streamlit interface for uploading documents and chatting with the model
- Local semantic search using `sentence-transformers/all-MiniLM-L6-v2`
- Token-based document chunking with retrieval and summary generation
- Local Gemma 4 E4B IT Q4_K_M quantized GGUF model served through `llama.cpp`
- Two response modes:
  - Precise for strict, document-grounded answers
  - Insightful for more exploratory answers that still stay evidence-aware
- Containerized deployment with Docker or Podman
- Jenkins pipeline included for automated container build and deployment

## How It Works

1. The user uploads a document and types a question.
2. The interface extracts text from the file.
3. The text is chunked by tokens and embedded locally.
4. Relevant chunks are retrieved from the in-memory index.
5. A whole-document summary is added to the final prompt.
6. The prompt is sent to the local `llama-server` container.
7. The model returns a grounded answer in Precise or Insightful mode.

## Project Structure

- `interface/` Streamlit app, document extraction, chunking, embeddings, retrieval, and prompt assembly
- `ml_backend/` `llama.cpp` backend image and model runtime
- `docker-compose.yml` local two-service orchestration
- `Jenkinsfile` Podman-based build and deployment pipeline

## Requirements

- Docker or Podman
- Docker Compose or Podman Compose
- Internet access on the first run so the model images and embedding model can be downloaded
- Enough RAM for a local LLM workflow <span style="color: red;">*(I recommend a practical minimum of about **16 GB RAM**)*</span>

## Run Locally

From the repository root:

```powershell
docker compose up --build
```

or, if you are using Podman:

```powershell
podman compose up --build
```

Open the app in your browser at:

```text
http://localhost:8501
```

The model server is exposed on:

```text
http://localhost:8082
```

To stop the stack:

```powershell
docker compose down
```

or:

```powershell
podman compose down
```

## How To Use

1. Start the compose stack.
2. Open the Streamlit UI.
3. Upload a `.txt`, `.pdf`, or `.docx` file.
4. Choose `Precise` or `Insightful`.
5. Ask a question about the document.

Precise mode is best for direct factual questions. Insightful mode is better when you want overlooked details, possible implications, risks, or things someone might miss.

## Models And Runtime

- Backend model: Gemma 4 E4B IT Q4_K_M quantized GGUF, served by `llama.cpp`
- Interface embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Backend context size: `10240`
- Output length: controlled by `n_predict` from the Streamlit app
- Prompting is document-grounded and uses:
  - whole-document summaries
  - retrieved chunks
  - mode-specific instructions for Precise and Insightful answers

## Coverage From The IBM-WUT MLOps Program

The project fulfills and supports the following ideas from the IBM MLOps program:

| Topic | How the project covers it |
| --- | --- |
| Agile methodology | The app has been refined iteratively, with incremental improvements to retrieval, summarization, and answer modes. |
| DevOps | The system is split into separate services for the interface and the model backend, both containerized and orchestrated together. |
| MLOps | The project includes model serving, document preprocessing, embedding-based retrieval, and repeatable deployment of the LLM stack. |
| AI lifecycle | It covers data preparation, model selection, deployment, and iterative refinement around the document QA workflow. |
| Continuous integration and delivery | The included `Jenkinsfile` automates environment cleanup, model deployment, and interface build/deploy steps with Podman. |
| Continuous testing and monitoring | The codebase is structured for extension, but a full monitoring stack and formal ML evaluation suite are not yet built in. |

The project does not follow the Waterfall model as its primary approach. It is built around iterative delivery and continuous refinement instead.

## Troubleshooting

- If the first startup is slow, the backend model and embedding model are probably being downloaded.
- If `docker compose` or `podman compose` fails, make sure the daemon or service is running.
- If answers are too slow, try `Precise` mode before `Insightful`.
- If a document does not seem searchable, verify that the file contains extractable text rather than only scanned images.

## Notes

- The interface keeps document state in memory for the current session.
- No external embedding API is required.
- The compose files use the Dockerfiles in `interface/` and `ml_backend/`.


## Teammates
- [Tănase Vlad Mihai](https://github.com/tanasevladmihai) - MLOps pipeline, model fine-tuning, RAG implementation, prompt engineering, model serving
- [Dimitriu Diandra](https://github.com/Diandra-Dimitriu) - Interface development, document processing workflow implementation, user experience design
- [Ștefanov Carla](https://github.com/carlastefanov05-cyber) - DevOps: Jenkins pipeline, containerization, deployment automation, testing, monitoring setup