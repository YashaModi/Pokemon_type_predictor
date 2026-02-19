.PHONY: clean data requirements train

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = pokemon_type_predictor
PYTHON_INTERPRETER = python3.11

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Make Dataset
data:
	$(PYTHON_INTERPRETER) pokemon_predictor/dataset.py

## Build Features
features:
	$(PYTHON_INTERPRETER) -m pokemon_predictor.build_features

## Train Models
train:
	$(PYTHON_INTERPRETER) -m pokemon_predictor.modeling.train

## Train Optimized MLP
train-opt:
	$(PYTHON_INTERPRETER) -m pokemon_predictor.modeling.train_optimized

## Train Hybrid MLP
train-hybrid:
	$(PYTHON_INTERPRETER) -m pokemon_predictor.modeling.train_hybrid

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
