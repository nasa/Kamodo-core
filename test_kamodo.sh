#!/bin/bash

PYTHONPATH=. coverage run -m pytest

coverage report -m

