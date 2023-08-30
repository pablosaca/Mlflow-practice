import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


__all__ = ["model_data_preprocessing",
           "metrics",
           "model_train",
           "preprocessing_features",
           "read_data"]


def model_data_preprocessing(config, datos):
    """
    Etapa de preprocesado previo al desarrollo del modelo (división de la muestra, one-hot enconding, etc)
    """
    # nombre de las columnas de interés
    columnas_corr = list(datos.select_dtypes(exclude=['category', 'object']).columns)
    corr_list = columnas_corr.copy()
    corr_list.remove(config.target.name)
    # Partición de la muestra
    X_train, X_test, y_train, y_test = train_test_split(datos.drop(columns=[config.target.name]),
                                                        datos[config.target.name],
                                                        test_size=config.modelling.train_test_sample.test_size,
                                                        random_state=config.modelling.train_test_sample.random_seed,
                                                        stratify=datos[config.target.name])  # división estratificada
    # Se convierten a variables dummy las no numéricas
    category_columns = list(datos.select_dtypes(include=['category', 'object']).columns)
    X_train_dummy = pd.get_dummies(X_train[category_columns],
                                   drop_first=config.modelling.ohe_task.drop_first,
                                   dtype=float)
    # Obtenemos las variables numéricas y concatenamos:
    X_train_num = X_train[corr_list]
    X_train = pd.concat([X_train_num, X_train_dummy], axis=1)  # se añaden a nivel registro

    # conversión a vector numpy la serie de pandas
    y_train = np.array(y_train)

    X_train, scale = scale_method(config, X_train)
    return X_train, y_train, scale


def scale_method(config, X_train):
    """
    Normalización o estandarización de las variables numéricas
    """
    if config.features.scale_method == "normalization":
        scale = MinMaxScaler()
        X_train_sc = scale.fit_transform(X_train)
        X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)

    elif config.features.scale_method == "standardization":
        scale = StandardScaler()
        X_train_sc = scale.fit_transform(X_train)
        X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)
    else:
        scale = None
    return X_train, scale


def metrics(config, model, X_train, y_train):
    """
    Métricas de evaulación (f1-score) en la muestra de entrenamiento
    """

    # F1-score (train)
    logging.info(f"f1-score con {config.modelling.threshold}")
    y_pred = np.where(model.predict_proba(X_train)[:, 1] >= config.modelling.threshold,
                      config.target.pos_class, config.target.neg_class)

    f1 = f1_score(y_train, y_pred)
    acc = accuracy_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)

    logging.info(f"f1 (train data): {f1}")
    logging.info(f"accuracy (train data): {acc}")
    logging.info(f"recall (train data): {recall}")
    logging.info(f"precision (train data): {precision}")

    return f1, acc, recall, precision


def model_train(config, X_train, y_train):
    """
    Entrenamiento del modelo con cross-validation
    """

    # El entrenamiento es realizado usando la técnica de Cross-Validation:
    # cross_validation
    cv = RepeatedStratifiedKFold(n_splits=config.modelling.training_process.n_splits,
                                 n_repeats=config.modelling.training_process.n_repeats,
                                 random_state=config.modelling.train_test_sample.random_seed)

    if config.modelling.model_type == "random_forest":

        rf = RandomForestClassifier(criterion=config.modelling.hyperparameters.random_forest.criterion,
                                    min_samples_split=config.modelling.hyperparameters.random_forest.min_samples_split,
                                    random_state=config.modelling.train_test_sample.random_seed)  # modelo base

        hyperameters = config.modelling.hyperparameters.random_forest
        HYPERPARAMETER_DICT = dict(hyperameters.copy())

        del HYPERPARAMETER_DICT["criterion"]
        del HYPERPARAMETER_DICT["min_samples_split"]

        # definición de los hiperparámetros como grid
        grid = [HYPERPARAMETER_DICT]

        # definición del modelo con hiperparámetros
        gs_model = GridSearchCV(estimator=rf,
                                param_grid=grid,
                                scoring=config.modelling.metric,
                                cv=cv,
                                n_jobs=config.modelling.training_process.n_jobs)

        # entrenamiento del modelo con hiperparámetros
        gs_model = gs_model.fit(X_train, y_train)

        for params, mean_metric in zip(gs_model.cv_results_["params"], gs_model.cv_results_["mean_test_score"]):
            logging.info(f"Hyperparams: {params}: {mean_metric}")

        # Se obtiene el mejor modelo
        model = gs_model.best_estimator_

        # Se analiza la bondad de ajuste con la muestra de entrenamiento (completa) y los datos de test.
        model.fit(X_train, y_train)

        model = CalibratedClassifierCV(model, cv=cv)
        model.fit(X_train, y_train)

    elif config.modelling.model_type == "logistic":
        model = LogisticRegression(max_iter=config.modelling.hyperparameters.logistic.max_iter)
        model.fit(X_train, y_train)

    else:
        model = None
    return model


def preprocessing_features(config, datos):
    """
    Preprocesado inicial de los datos. Reemplaza niveles de variables categóricas
    """

    # `workclass`
    cond = [datos["workclass"].isin(["Self-emp-not-inc", "Self-emp-inc"]),
            datos["workclass"].isin(["State-gov", "Local-gov", "Federal-gov"]),
            datos["workclass"].isin(["?", "Without-pay", "Never-worked"])]
    values = ["Self", "State", "Other"]
    datos["workclass"] = np.select(cond, values, default=datos["workclass"])
    datos["workclass"] = datos["workclass"].astype("category")

    # `education`
    cond = [datos["education"].isin(["Assoc-acdm", "Assoc-voc"]),
            datos["education"].isin(["Preschool", "HS-grad"]),
            datos["education"].isin(["Bachelors", "Masters", "Doctorate", "Some-college"]),
            datos["education"] == "Prof-school"]
    values = ["Associate", "School", "BCMD", "Prof-school"]
    datos["education"] = np.select(cond, values, default="Xth")
    datos["education"] = datos["education"].astype("category")

    # `marital-status`
    cond = [datos["marital-status"].isin(["Separated", "Divorced"]),
            datos["marital-status"].isin(["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"]),
            datos["marital-status"] == "Widowed"]
    values = ["Divorced/Separated", "Married", "Widowed"]
    datos["marital-status"] = np.select(cond, values, default="Never-married")
    datos["marital-status"] = datos["marital-status"].astype("category")

    # `occupation`
    values = ["Protective-serv", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales", "Machine-op-inspct"]
    datos["occupation"] = np.where(datos["occupation"].isin(values), datos["occupation"], "Other")
    datos["occupation"] = datos["occupation"].astype("category")

    # define target values
    datos[config.target.name] = np.where(datos[config.target.name] == ">50K",
                                         config.target.pos_class, config.target.neg_class)

    return datos


def read_data(config, data_format, data_path):
    """
    Lectura de datos
    """
    if data_format != "csv":
        datos = pd.DataFrame()
    else:
        # Se leen los datos. Y un pequeño resumen de las variables numéricas
        datos = pd.read_csv(data_path).drop(columns=config.features.drop_features)
    return datos
