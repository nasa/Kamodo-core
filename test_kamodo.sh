#!/bin/bash

PYTHONPATH=. coverage run -m pytest tests/test_kamodo.py

coverage report -m

