pipeline {
    agent any
    stages {
        stage('Clean Environment') {
        steps {
                sh 'sudo podman rm -f llama-server interface-server || true'
                sh 'sudo podman network create llm-net || true'
            }
        }
        stage('Deploy ML Model (llama.cpp)') {
        steps {
                sh 'sudo podman run -d --name llama-server --network llm-net --memory=4g -v /opt/models:/models ghcr.io/ggml-org/llama.cpp:server -m /models/gemma-4-e4b.gguf --host 0.0.0.0 --port 8080 --threads 4 --ctx-size 2048'
            }
        }
        stage('Build & Deploy Interface') {
            steps {
               dir('interface') {
                    sh 'sudo podman build -t streamlit-app .'
                    sh 'sudo podman run -d --name interface-server --network llm-net -p 8501:8501 streamlit-app'
               }
            }

        }
    }
}