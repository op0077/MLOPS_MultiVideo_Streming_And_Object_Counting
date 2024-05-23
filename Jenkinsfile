pipeline{
    agent any
    
    stages{
        stage('Git checkout'){
            steps{
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/Harshil00077/Object_Counting.git']])
            }
        }
        stage('Containers Down'){
            steps{
                script{
                    sh 'docker-compose -f kafka/docker-compose.yml down'
                    sh 'docker-compose -f docker-compose.yml down'
                }
            }
        }
        stage('Remove'){
            steps{
                script{
                    sh 'git rm -r --cached "torchserve/models/detection.onnx"'
                    sh 'git rm -r --cached "torchserve/models/trained_model_10epoch.onnx"'
                    sh 'git rm -r --cached "torchserve/models/trained_model_60epoch.onnx"'
                }
            }
        }
        // stage('Containers Up'){
        //     steps{
        //         script{
        //             sh 'docker-compose -f kafka/docker-compose.yml up -d'
        //             sh 'docker-compose -f docker-compose.yml up -d'
        //         }
        //     }
        // }
        
       
        
        // stage('Run Ansible playbook'){
        //     steps{
        //     	script{
        //     	    ansiblePlaybook(playbook:'ansible/deploy.yml',inventory:'ansible/inventory')
        //     	}
        //     }
        // }
    }
}
