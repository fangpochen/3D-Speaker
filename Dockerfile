# 使用官方Python镜像 - 升级到3.9支持现代包
FROM python:3.9-slim

# 设置构建时代理 - 通过build-arg传入
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG no_proxy
ARG NO_PROXY

# 使代理参数在构建时可用作环境变量
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV no_proxy=${no_proxy}
ENV NO_PROXY=${NO_PROXY}

WORKDIR /app

# 安装系统依赖（增强重试机制与错误处理）
RUN apt-get update && \
    # 设置APT的重试次数和超时时间
    echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::http::Timeout "180";' >> /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::https::Timeout "180";' >> /etc/apt/apt.conf.d/80retries && \
    # 每个包分开安装，以便单个包失败不影响整体
    apt-get install -y --no-install-recommends --fix-missing build-essential && \
    apt-get install -y --no-install-recommends --fix-missing libsndfile1 && \
    apt-get install -y --no-install-recommends --fix-missing ffmpeg && \
    # 清理APT缓存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖文件
COPY requirements.txt .

# 创建一个CPU版的要求文件
RUN grep -v torch requirements.txt | grep -v torchaudio > requirements_cpu.txt

# 安装Python依赖（添加重试机制）- 使用CPU版本的PyTorch和torchaudio
RUN pip install --retries 10 --timeout 180 -r requirements_cpu.txt && \
    pip install --retries 10 --timeout 180 torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --retries 10 --timeout 180 torchaudio==0.13.1+cpu --index-url https://download.pytorch.org/whl/cpu

# 先复制speakerlab包，因为它是本地依赖
COPY speakerlab ./speakerlab/

# 后复制应用代码（这样修改代码不会导致重新安装依赖）
COPY voice_auth_app.py .
COPY download_model.py .

# 创建模型和数据目录
RUN mkdir -p /app/modelscope_hub_cache /app/voice_auth_db/embeddings /app/voice_auth_db/tts_audio

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODELSCOPE_CACHE=/app/modelscope_hub_cache
ENV PYTHONPATH="${PYTHONPATH}:/app/speakerlab"

# 不在镜像中硬编码代理地址，而是在运行时通过docker-compose.yml提供
# 添加no_proxy设置，避免内部通信也走代理
ENV no_proxy=localhost,127.0.0.1
ENV NO_PROXY=localhost,127.0.0.1

# 开放端口
EXPOSE 7860

# 启动应用
CMD ["python", "voice_auth_app.py"] 