# ベースイメージとしてUbuntu 22.04を使用
FROM ubuntu:22.04

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Minicondaをダウンロードしてインストール
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# condaコマンドを環境変数に追加
ENV PATH /opt/conda/bin:$PATH

# condaの初期設定を実行
RUN conda update -n base -c defaults conda

# 必要なPythonパッケージをインストール
RUN conda install -y python=3.9 && \
    conda install -y \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    notebook && \
    conda install -y -c pytorch \
    pytorch \
    torchvision && \
    conda clean -ya

# 作業ディレクトリを作成
WORKDIR /workspace

# Jupyter Notebookの設定ファイルを作成
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py

# コンテナ起動時にJupyter Notebookを実行
CMD ["jupyter", "notebook", "--allow-root"]

# 必要なポートを公開
EXPOSE 8888
