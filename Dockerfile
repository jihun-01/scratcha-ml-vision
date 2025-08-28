# TensorFlow GPU 기반 이미지 사용
FROM tensorflow/tensorflow:2.13.0-gpu

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정 (선택사항 - 필요에 따라 수정)
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=0

# 포트 노출 (필요한 경우)
EXPOSE 8080

# 헬스체크 추가 (선택사항)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import tensorflow as tf; print('TensorFlow 버전:', tf.__version__)" || exit 1

# 애플리케이션 실행
CMD ["python", "quiz/object_detection_quiz_generator.py"]

