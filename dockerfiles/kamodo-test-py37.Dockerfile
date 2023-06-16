FROM ensemble/kamodo-py37

# RUN python -m pip install --upgrade pip
RUN python -m pip install flake8 pytest
RUN pip install pytest-cov
RUN pip install requests
RUN pip install sympy==1.5.1
RUN pip install notebook

COPY . /kamodo
WORKDIR /


SHELL  ["sh", "-c", "chmod +x /kamodo/test_kamodo.sh"]

WORKDIR /kamodo

CMD test_kamodo.sh