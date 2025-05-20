FROM python:3.10-slim

# 必要な依存パッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    scikit-learn \
    jupyter

# 作業ディレクトリ作成
WORKDIR /workspace

# Jupyter Notebookの起動
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

EXPOSE 8888
