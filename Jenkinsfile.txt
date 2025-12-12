pipeline {
    agent any

    environment {
        PYTHON = "C:\\Program Files\\Python314\\python.exe"
    }

    stages {

        stage('Initialize') {
            steps {
                script {
                    // Upgrade pip and install necessary Python packages
                    bat "\"${env.PYTHON}\" -m pip install --upgrade pip"
                    bat "\"${env.PYTHON}\" -m pip install -r requirements.txt"
                }
            }
        }

        stage('Load and Preprocess Data') {
            steps {
                script {
                    // Run data loading script
                    bat "\"${env.PYTHON}\" data_preprocessing_exploration.py"
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    // Run model training script
                    bat "\"${env.PYTHON}\" model_training.py"
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    // Run model evaluation script
                    bat "\"${env.PYTHON}\" model_evaluation.py"
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/*.pkl', fingerprint: true
            echo 'Pipeline execution complete.'
        }
    }
}
