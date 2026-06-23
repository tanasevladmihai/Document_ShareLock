pipeline {
    agent any
    //Prevent the pipeline from hanging or overlapping
    options {
        timeout(time: 30, unit: 'MINUTES') // Kills the job if it freezes
        disableConcurrentBuilds()          // Prevents two deployments from running at the same time
    }
    
    stages {
        stage('Deep Clean & Prep') {
            steps {
                // Using 'sudo' since your previous pipeline required administrator privileges
                sh 'sudo podman compose down -v --remove-orphans || true'
                
                // Wipe out hidden cache layers to prevent the "No space left on device" error
                sh 'sudo podman system prune -a -f --volumes'
            }
        }
        
        stage('Build & Deploy Containers') {
            steps {
                // Let Compose handle the build, network routing, and startup
                sh 'sudo podman compose up --build -d'
            }
        }

        stage('Verify Health') {
            steps {
                // Wait 15 seconds to give the containers time to crash if there is a critical error
                sleep 15
                
                // Print the running containers to the Jenkins log for easy debugging
                sh 'sudo podman ps'
                
                // Ping the Streamlit interface. If it doesn't answer, fail the pipeline immediately.
                sh 'curl -f -I http://localhost:8501 || exit 1'
            }
        }
    }
    
    // Auto-fetch logs if something breaks
    post {
        failure {
            echo 'Deployment failed! Fetching the crash logs...'
            sh 'sudo podman compose logs'
        }
        success {
            echo 'Deployment successful and health checks passed!'
        }
    }
}
