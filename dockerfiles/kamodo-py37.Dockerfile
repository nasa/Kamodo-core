# docker build -t asherp/kamodo -f API.Dockerfile .
FROM condaforge/miniforge3
# FROM continuumio/miniconda3:latest
LABEL maintainer "Asher Pembroke <apembroke@gmail.com>"

RUN conda install python=3.7

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

# capnproto pip version
RUN conda install gcc cmake make cxx-compiler
RUN pip install pkgconfig cython

# # install release
RUN  wget https://capnproto.org/capnproto-c++-0.9.1.tar.gz
RUN  tar zxf capnproto-c++-0.9.1.tar.gz
WORKDIR capnproto-c++-0.9.1
RUN  ./configure 
RUN  make -j6 check
RUN  make install


WORKDIR /

RUN git clone https://github.com/capnproto/pycapnp.git
WORKDIR /pycapnp

RUN python setup.py install --force-bundled-libcapnp

RUN git clone --single-branch --branch rpc https://github.com/EnsembleGovServices/kamodo-core.git

RUN pip install -e kamodo-core
WORKDIR /kamodo-core/kamodo/rpc
CMD python test_rpc_kamodo_server.py 
