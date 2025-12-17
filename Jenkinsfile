pipeline {
    agent any

    environment {
        IMAGE_NAME  = "ocr_tg_bot"
        AI_API_BASE = "http://net.unequaled-earthquake.ru:5555"
        NAME_SUFFIX = '-dev'
        KEEP_BUILDS = "10"
    }

    parameters {
        booleanParam(name: 'RUN_ML_IMPORT', defaultValue: false, description: 'Запустить ML batch import (долго!)')
        booleanParam(name: 'RUN_ML_TRAIN',  defaultValue: false, description: 'Запустить ML train models (долго!)')
    }

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 30, unit: 'MINUTES')  // Общий таймаут на весь pipeline
    }

    stages {
        stage('Cleanup old images') {
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                sh '''
                set +e
                echo "=== Удаляю dangling images ==="
                docker image prune -f
                
                echo "=== Удаляю неиспользуемые контейнеры и volumes ==="
                docker container prune -f
                docker volume prune -f
                
                echo "=== Удаляю старые теги ocr_tg_bot (оставляю последние ${KEEP_BUILDS}) ==="
                TAGS=$(docker images --format "{{.Tag}}" ${IMAGE_NAME} | grep -E '^[0-9]+$' | sort -rn)
                COUNT=0
                for TAG in $TAGS; do
                  COUNT=$((COUNT + 1))
                  if [ $COUNT -gt ${KEEP_BUILDS} ]; then
                    echo "Удаляю ${IMAGE_NAME}:${TAG}"
                    docker rmi "${IMAGE_NAME}:${TAG}" || true
                  fi
                done
                
                set -e
                docker system df
                '''
            }
        }

        stage('Checkout') {
            steps {
                deleteDir()
                git branch: 'dev', url: 'https://github.com/mrbastim/ocr_tg.git'
            }
        }

        stage('Build Docker image') {
            options { timeout(time: 15, unit: 'MINUTES') }
            steps {
                sh 'docker build --progress=plain -t ${IMAGE_NAME}:${BUILD_NUMBER} .'
            }
        }

        stage('Smoke tests') {
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                sh '''
                docker run --rm \
                  -e AI_API_BASE=${AI_API_BASE} \
                  -e AI_API_BASE_PATH=/api \
                  ${IMAGE_NAME}:${BUILD_NUMBER} \
                  bash -c "pytest -q tests/test_backend_api.py && python -m compileall ."
                '''
            }
        }

        stage('ML batch import') {
            when { 
                expression { params.RUN_ML_IMPORT == true }
            }
            options { timeout(time: 60, unit: 'MINUTES') }
            steps {
                sh '''
                set -e
                echo "=== Запускаю пакетный импорт (может занять до часа) ==="
                docker run --rm \
                  -v /root/ocr_tg/photos:/data \
                  -v ml_output:/app/ml_output \
                  -w /app \
                  ${IMAGE_NAME}:${BUILD_NUMBER} \
                  python -m ml.batch_import --dir /data --lang rus+eng --provider Jenkins --workers 6 --max-files 150
                '''
            }
        }

        stage('ML train models') {
            when { 
                expression { params.RUN_ML_TRAIN == true }
            }
            options { timeout(time: 60, unit: 'MINUTES') }
            steps {
                sh '''
                set -e
                if docker run --rm -v ml_output:/app/ml_output -w /app \
                     ${IMAGE_NAME}:${BUILD_NUMBER} \
                     test -f ml_output/events.csv ; then
                  echo "Найден ml_output/events.csv, запускаю обучение моделей..."
                  docker run --rm \
                    -v ml_output:/app/ml_output \
                    -w /app \
                    ${IMAGE_NAME}:${BUILD_NUMBER} \
                    python -m ml.train_models --csv ml_output/events.csv --target-col total_time
                else
                  echo "⚠️ Файл ml_output/events.csv не найден, этап обучения пропущен."
                fi
                '''
            }
        }

        stage('Deploy') {
            steps {
                withCredentials([
                    string(credentialsId: 'tg_bot_token', variable: 'TG_BOT_TOKEN')
                ]) {
                    sh '''
                    set -e

                    # Гарантируем существование тома для логов/моделей
                    docker volume create ml_output || true

                    # Генерируем .env для бота из секретов Jenkins
                    cat > bot/.env <<EOF
TELEGRAM_BOT_TOKEN=${TG_BOT_TOKEN}
AI_API_BASE=${AI_API_BASE}
AI_API_BASE_PATH=/api
AI_API_DEBUG=1
EOF

                    # Обновляем сервис через docker compose
                    docker compose down || true
                    docker compose up -d --build
                    '''
                }
            }
        }
    }

    post {
        always {
            echo "Build #${BUILD_NUMBER} finished with status: ${currentBuild.currentResult}"
            sh '''
            echo "=== Финальное состояние Docker ==="
            docker system df || true
            '''
        }
        failure {
            sh '''
            echo "=== Логи контейнеров при ошибке ==="
            docker compose logs --tail=50 || true
            '''
        }
    }
}
