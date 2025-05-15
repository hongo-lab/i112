FROM ubuntu:22.04

# 必要パッケージ
RUN apt-get update && apt-get install -y \
    wget bzip2 ca-certificates curl git libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Miniforge (x86_64/arm64 自動対応)
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh ; \
    else \
      wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh ; \
    fi && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Pythonと必要パッケージ
RUN conda update -n base -c defaults conda && \
    conda install -y python=3.9 numpy pandas scikit-learn matplotlib jupyter notebook && \
    conda install -y -c pytorch pytorch torchvision && \
    conda clean -ya

WORKDIR /workspace

# Jupyter設定
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py

CMD ["jupyter", "notebook", "--allow-root"]
EXPOSE 8888
