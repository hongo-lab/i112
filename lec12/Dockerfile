FROM ubuntu:22.04

# 必要なシステムパッケージとPythonのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pipのアップグレード
RUN pip install --upgrade pip

# Pythonパッケージのインストール
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.3 \
    notebook==7.2.1

# PyTorch CPU版のインストール
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace

# Jupyterの設定（notebook 7.xはServerAppを使用）
RUN jupyter notebook --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root"]

