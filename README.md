# Tutorial Mlflow

This repo consist of a basic guide to use MLOps package to register, manage and deploy an ML project

## Installation

Create a conda virtual environment in the project directory.

```
conda create -n mlops_env python=3.9
```

Activate the virtual environment.

```
conda activate mlops_env
```

While in the virtual environment, install required dependencies from `requirements.txt`.

```
pip install -r requirements.txt
```

To start mlflow functionality it is necessary to use one of the following commands

```
mlflow server
```
```
mlflow ui
```

The application may then be terminated with the following commands.

```
ctrl - c
```

## Project Structure 

```
├── config
├── data
├── utils
│   ├── utils.py
├── train_model.py
├── requirements.txt
└── README.md
```
