#!/bin/bash

pytest --cov kamodo.kamodo --cov kamodo.util --cov plotting kamodo/test_plotting.py kamodo/test_kamodo.py kamodo/test_utils.py kamodo/test_rpc.py

jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True --config docs/notebooks/run_notebooks.py