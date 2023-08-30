import logging
import warnings

import hydra
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from numpy.random import seed
from hydra import utils
from omegaconf.dictconfig import DictConfig

from utils.utils import (
    model_data_preprocessing,
    metrics,
    model_train,
    preprocessing_features,
    read_data
)


@hydra.main(config_path="config", config_name="config_file.yaml", version_base="1.1")
def run_training(config: DictConfig):
    """
    Método general. Entrenamiento de un modelo de Machine Learning
    Uso de ficheros de configuración para realizar el proceso de aprendizaje
    """
    warnings.filterwarnings("ignore")

    # Inicia un experimento de MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  # URL del servidor MLflow
    mlflow.set_experiment(experiment_id=config.mlflow_register.experiment_id)

    with mlflow.start_run():

        logging.info("EMPEZAMOS...")
        seed(config.default.seed)  # default seed for numpy

        # path del código fuente
        current_path = utils.get_original_cwd()
        current_path = current_path.replace(r"\src", "")

        # directorio, nombre y formato donde se encuentran los datos
        data_path = config.dataset.path
        data_name = config.dataset.data
        data_format = config.dataset.format

        data_path = f"{current_path}/{data_path}/{data_name}.{data_format}"
        datos = read_data(config, data_format, data_path)
        if datos.empty:
            logging.info("Format file is not `csv`")
            raise TypeError("Format file is not `csv`")

        datos = preprocessing_features(config, datos)

        available_scale_method = ["normalization", "standardization"]
        X_train, y_train, scale = model_data_preprocessing(config, datos)
        if scale is None:
            logging.info(f"Incorrect `scale` method. Only available {available_scale_method}")
            raise ValueError(f"Incorrect `scale` method. Only available {available_scale_method}")

        model = model_train(config, X_train, y_train)

        available_model_type = ["random_forest", "logistic"]
        if model is None:
            logging.info(f"Incorrect model name. Only available {available_model_type}")
            raise ValueError(f"Incorrect model name. Only available {available_model_type}")

        f1, acc, recall, precision = metrics(config, model, X_train, y_train)
        # se crea un diccionario con las métricas
        metrics_dict = {"f1": f1, "acc": acc, "recall": recall, "precision": precision}

        logging.info("Registro auxiliares: features, parámetros, métricas")

        # nombre de las features
        mlflow.set_tag("features", list(scale.feature_names_in_))

        # parámetros de muestras entrenamiento/validación
        mlflow.log_param("test_size", config.modelling.train_test_sample.test_size)
        mlflow.log_param("seed", config.modelling.train_test_sample.random_seed)

        # parámetros del modelo
        mlflow.log_params(model.get_params())

        # métricas del modelo
        mlflow.log_metrics(metrics_dict)

        logging.info("Registro del dataset de entrenamiento")
        dataset = mlflow.data.from_pandas(X_train, source=config.dataset.data)
        mlflow.log_input(dataset=dataset, context="training_dataset")

        logging.info("Registro del modelo: escalado y modelo")
        # incluye signature que es el tipado de los datos a la hora de guardar los modelos
        model_signature = infer_signature(model_input=X_train, model_output=model.predict(X_train))
        scale_signature = infer_signature(model_input=X_train, model_output=scale.transform(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=config.mlflow_register.path_model_name,
            registered_model_name=config.mlflow_register.registered_model_name,
            signature=model_signature)  # modelo

        mlflow.sklearn.log_model(
            sk_model=scale,
            artifact_path=config.mlflow_register.path_scale_name,
            registered_model_name=config.mlflow_register.registered_scale_name,
            signature=scale_signature)  # escalado

        logging.info("PROCESO TERMINADO. REGISTRADO EL MODELO EN MLFLOW...")


if __name__ == '__main__':

    run_training()

