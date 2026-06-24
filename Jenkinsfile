pipeline {
    agent any

    stages {
        stage('Initialize') {
            steps {
                script {
                    def configuredModelCacheDir = env.MODEL_CACHE_DIR?.trim()
                    env.MODEL_CACHE_DIR = configuredModelCacheDir ?: '/opt/document-sharelock/models'
                }
                echo "Using model cache directory: ${env.MODEL_CACHE_DIR}"
            }
        }

        stage('Preflight') {
            steps {
                sh '''
                    set -eu

                    test -f docker-compose.yml
                    test -f ml_backend/Dockerfile
                    test -f interface/Dockerfile

                    command -v podman
                    sudo podman --version
                    sudo podman compose version
                '''
            }
        }

        stage('Prepare Model Cache') {
            steps {
                sh '''
                    set -eu

                    sudo mkdir -p "${MODEL_CACHE_DIR}"
                    sudo chmod 775 "${MODEL_CACHE_DIR}" || true
                '''
            }
        }

        stage('Stop Existing Stack') {
            steps {
                sh '''
                    set -eu

                    sudo env MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" podman compose down --remove-orphans
                '''
            }
        }

        stage('Build Stack') {
            steps {
                sh '''
                    set -eu

                    sudo env MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" podman compose build --pull
                '''
            }
        }

        stage('Deploy Stack') {
            steps {
                sh '''
                    set -eu

                    sudo env MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" podman compose up -d
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                sh '''
                    set -eu

                    for container in llama-server interface-server prometheus grafana; do
                        sudo podman container exists "${container}"
                    done

                    for container in llama-server interface-server; do
                        healthy=""
                        for attempt in $(seq 1 60); do
                            status="$(sudo podman inspect "${container}" --format '{{.State.Health.Status}}')"
                            if [ "${status}" = "healthy" ]; then
                                healthy="true"
                                break
                            fi

                            if [ "${status}" = "unhealthy" ]; then
                                echo "${container} became unhealthy."
                                sudo podman logs --tail 100 "${container}" || true
                                exit 1
                            fi

                            sleep 10
                        done

                        if [ "${healthy}" != "true" ]; then
                            echo "${container} did not become healthy in time."
                            sudo podman logs --tail 100 "${container}" || true
                            exit 1
                        fi
                    done

                    for container in prometheus grafana; do
                        running="$(sudo podman inspect "${container}" --format '{{.State.Running}}')"
                        if [ "${running}" != "true" ]; then
                            echo "${container} is not running."
                            sudo podman logs --tail 100 "${container}" || true
                            exit 1
                        fi
                    done

                    memory_limit="$(sudo podman inspect llama-server --format '{{.HostConfig.Memory}}')"
                    if [ "${memory_limit}" != "0" ] && [ "${memory_limit}" != "<no value>" ] && [ -n "${memory_limit}" ]; then
                        echo "llama-server has an unexpected memory limit configured: ${memory_limit}"
                        exit 1
                    fi

                    sudo env MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" podman compose ps
                '''
            }
        }
    }

    post {
        always {
            sh '''
                sudo env MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/opt/document-sharelock/models}" podman compose ps || true
            '''
        }
    }
}
