# docker build -t asherp/kamodo -f API.Dockerfile .

FROM continuumio/miniconda3:latest
LABEL maintainer "Asher Pembroke <apembroke@gmail.com>"

RUN conda install -c conda-forge python=3.6 pip

# RUN conda install jupyter
RUN pip install antlr4-python3-runtime


# # Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
# ENV TINI_VERSION v0.6.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini", "--"]

# need to pin this version for api
RUN pip install sympy==1.5.1

# Keep plotly at lower api
RUN pip install plotly==4.7.1

# kaleido for generating static plots
RUN pip install kaleido

RUN pip install pycapnp

# jupyter
RUN conda install -c conda-forge jupyter
RUN pip install jupytext

# Install latest kamodo
ADD . /kamodo

WORKDIR /kamodo

# # RUN git clone https://github.com/asherp/kamodo.git
RUN pip install .


# # CMD ["kamodo-serve"]

CMD ["jupyter", "notebook", "./docs/notebooks", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]


