FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL authors="jaret"

# Чтобы apt не задавал вопросы про timezone
ENV DEBIAN_FRONTEND=noninteractive

# (опционально) актуальные архитектуры для компиляции CUDA-опов
# Если что-то компилируется из исходников (flash-attn и т.п.)
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# Базовые зависимости
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    tmux \
    htop \
    nvtop \
    python3-opencv \
    openssh-client \
    openssh-server \
    openssl \
    rsync \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Node.js (LTS достаточно для Next.js UI)
WORKDIR /tmp
RUN curl -sL https://deb.nodesource.com/setup_20.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Рабочая директория проекта
WORKDIR /app/ai-toolkit

# Копируем весь репозиторий внутрь контейнера
# ВАЖНО: билд запускать из корня: docker build -f docker/Dockerfile .
COPY . .

# Алиасы для python / pip
RUN ln -s /usr/bin/python3 /usr/bin/python || true && \
    ln -s /usr/bin/pip3 /usr/bin/pip || true

# Сначала ставим PyTorch (как в README ai-toolkit: torch 2.7.0 + cu126)
# cu126 отлично работает с драйвером/рантаймом 12.8
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Остальные Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# (опционально) если понадобятся подмодули, раскомментируй:
# RUN git submodule update --init --recursive

# Cборка UI
WORKDIR /app/ai-toolkit/ui

RUN npm install && \
    npm run update_db && \
    npm run build

# Порт UI
EXPOSE 8675

# Стартовый скрипт (он уже есть в репо в папке docker)
WORKDIR /
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
