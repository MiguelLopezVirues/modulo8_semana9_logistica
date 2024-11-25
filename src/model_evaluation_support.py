# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Para realizar la regresión lineal y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_validate

from sklearn.feature_selection import RFECV
from skopt import BayesSearchCV

import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


# Gestionar los warnings
# -----------------------------------------------------------------------
import warnings

# modificar el path
# -----------------------------------------------------------------------
import sys
import os
sys.path.append("..")

# crear archivos temporales
# -----------------------------------------------------------------------
import tempfile

# statistics functions
# -----------------------------------------------------------------------
from scipy.stats import  norm


# Registro de modelos
# -----------------------------------------------------------------------
import mlflow
import mlflow.sklearn


# Acceder a parámetros de métodos de objetos
# -----------------------------------------------------------------------
import inspect


seed = 42


def model_evaluation_CV_run(run_name, models, scores, X, y, crossval, verbose=False):
    results = []

    n_splits = crossval.get_n_splits(X=X, y=y)

    warnings.filterwarnings('ignore')
    mlflow.autolog()

    with mlflow.start_run(run_name=f"{run_name} - Model Evaluation CV"):
        for name, model in models:
            # Start a child run for each model
            with mlflow.start_run(run_name=f"{name} - Model Evaluation CV", nested=True):

                mlflow.log_param("model_name", name)
                mlflow.log_param("cross_val_splits", n_splits)

                if verbose:
                    print(f"\nTraining {name}.")
                # Cross_val
                cv_results = cross_validate(model, X, y, cv=crossval, scoring=scores, verbose=verbose, return_train_score=True)

                # Store results for each fold and each metric
                for split in range(n_splits):
                    result = {"Model": name, "Split": split + 1}
                    for score in scores:
                        # prepare results_df
                        result[f"test_{score}"] = cv_results[f"test_{score}"][split]
                        result[f"train_{score}"] = cv_results[f"train_{score}"][split]

                    # log metrics
                    mlflow.log_metric(f"mean_test_{score}", cv_results[f"test_{score}"].mean())
                    mlflow.log_metric(f"std_test_{score}", cv_results[f"test_{score}"].std())
                    mlflow.log_metric(f"mean_train_{score}", cv_results[f"train_{score}"].mean())
                    mlflow.log_metric(f"std_train_{score}", cv_results[f"train_{score}"].std())
                    results.append(result)
                

                if verbose:
                    # Print results numerically
                    print(f'--\n{name} model:')
                    for score in scores:
                        print("%s: mean %f, std (%f) " % (score, cv_results[f"test_{score}"].mean(), cv_results[f"test_{score}"].std()))


        results_df = pd.DataFrame(results)
        log_dataframe_to_mlflow(results_df, artifact_path="test_results")

    warnings.filterwarnings('default')


    return results_df


def run_gridsearch_experiment(X_train, y_train, model_name, model, param_grid, cross_val, score, verbosity):
    mlflow.autolog()
    with mlflow.start_run(run_name=f"GridSearch_CV_{model_name}"):

        # Dynamically set verbosity if supported
        if 'verbose' in inspect.signature(model.__init__).parameters:
            model.verbose = verbosity  

        grid_search = GridSearchCV(model, 
                                param_grid, 
                                cv=cross_val, 
                                scoring=score, 
                                n_jobs = -1,
                                verbose=verbosity)

        grid_search.fit(X_train, y_train)
    return {
        "model_name": model_name,
        "pipeline": grid_search.best_estimator_,
        "params": grid_search.best_params_,
        "score": grid_search.best_score_
    }


def run_bayessearch_experiment(X_train, y_train, model_name, model, param_space, cross_val, n_iter, score,verbosity, seed =42):
    mlflow.autolog()
    with mlflow.start_run(run_name=f"BayesSearch_CV_{model_name}"):

        # Dynamically set verbosity if supported
        if 'verbose' in inspect.signature(model.__init__).parameters:
            model.verbose = verbosity  
        
        bayes_search = BayesSearchCV(estimator=model,
                                    search_spaces=param_space, 
                                    n_iter=n_iter, 
                                    cv=cross_val, 
                                    scoring=score,
                                    n_jobs=-1,
                                    random_state=seed)

        bayes_search.fit(X_train, y_train)
    return {
        "model_name": model_name,
        "pipeline": bayes_search.best_estimator_,
        "params": bayes_search.best_params_,
        "score": bayes_search.best_score_
    }



def plot_prediction_residuals(y_test, y_test_pred):

    fig, axes = plt.subplots(2,2, figsize=(15,8))
    axes = axes.flat

    sns.histplot(y_test_pred, bins=30, ax=axes[0], label="y_test_pred")
    sns.histplot(y_test, bins=30, ax=axes[0], alpha=0.25, label="y_test")
    axes[0].legend()
    axes[0].set_title('y_test vs y_test_pred distribution')

    axes[1].scatter(y_test.reset_index(drop=True), y_test_pred, color='purple')

    axes[1].axline((0, 0), slope=1, color='r', linestyle='--')
    axes[1].set_title('y_test Vs. y_pred')
    axes[1].set_xlabel('True values')
    axes[1].set_ylabel('Predicted values')


    residuals = y_test.values - y_test_pred

    sns.histplot(residuals, bins=30, ax=axes[2])
    axes[2].set_title('Residuals distribution')

    # Plot residuals
    axes[3].scatter(y_test_pred, residuals, color='purple')
    axes[3].axhline(0, linestyle='--', color='black', linewidth=1)
    axes[3].set_title('Residuals Plot')
    axes[3].set_xlabel('Predicted Values')
    axes[3].set_ylabel('Residuals')



    plt.tight_layout()
    plt.show()


def test_evaluate_model(run_name, model, X_train, y_train, X_test, y_test, best_params=None, tag=None,train_multiplier=1,test_multiplier=1):
    mlflow.autolog()

    with mlflow.start_run(run_name=f"{run_name} - test_evaluation") as run:
        modelo = model.set_params(**(best_params or {}))

        modelo.fit(X_train, y_train)

        y_train_pred = modelo.predict(X_train) * train_multiplier
        y_test_pred = modelo.predict(X_test) * test_multiplier
        y_train = y_train*train_multiplier
        y_test = y_test*test_multiplier

        mlflow.autolog(disable=True)
        metricas = calculate_train_test_metrics(y_train, y_test, y_train_pred, y_test_pred)

        # Registrar metricas a MLflow
        for train_test in metricas.keys():
            for metric_name, value in metricas[train_test].items():
                mlflow.log_metric(f"{train_test}_{metric_name}", value)

        mlflow.autolog()

        resultados_df = pd.DataFrame(metricas).T

        # Guardar como csv para poder registrarlo en MLFlow
        log_dataframe_to_mlflow(resultados_df, artifact_path="test_results")

        # Guardar tag con detalles adicionales
        if tag:
            mlflow.set_tag(tag[0], tag[1])

        plot_prediction_residuals(y_test, y_test_pred)

    return resultados_df



def log_dataframe_to_mlflow(dataframe: pd.DataFrame, artifact_path: str):
    """
    Save a DataFrame as a temporary CSV file and log it as an artifact in MLflow.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be saved and logged.
    artifact_path : str
        The artifact path within MLflow where the file will be stored.
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp_results.csv")
        
        # Save DataFrame to the temporary file
        dataframe.to_csv(temp_file_path, index=False)
        
        # Log with MLflow
        mlflow.log_artifact(temp_file_path, artifact_path=artifact_path)
        



def calculate_train_test_metrics(y_train, y_test, y_train_pred, y_test_pred):
        # Calcular modelo naive de la media para train y test
        y_mean_train = np.mean(y_train)
        y_mean_pred_train = np.full_like(y_train, y_mean_train)

        y_mean_test = np.mean(y_test)
        y_mean_pred_test = np.full_like(y_test, y_mean_test)

        # Calcular todas las métricas
        metricas = {
            'train': {
                'r2_score': r2_score(y_train, y_train_pred),
                'MAE': mean_absolute_error(y_train, y_train_pred),
                'MSE': mean_squared_error(y_train, y_train_pred),
                'MSE_naive': mean_squared_error(y_train, y_mean_pred_train),
                'RMSE': root_mean_squared_error(y_train, y_train_pred)
            },
            'test': {
                'r2_score': r2_score(y_test, y_test_pred),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'MSE': mean_squared_error(y_test, y_test_pred),
                'MSE_naive': mean_squared_error(y_test, y_mean_pred_test),
                'RMSE': root_mean_squared_error(y_test, y_test_pred)

            }
        }

        return metricas


def calcular_ic(df: pd.DataFrame, columna: str, alpha: float, metodo: str ="normal", seed: int = None, n_bootstrap: int = 1000) -> tuple:
    """
    Calcula el intervalo de confianza al 95% para una columna de un DataFrame
    utilizando la distribución normal.
    
    Args:
        data (pd.DataFrame): Dataset con los datos.
        columna (str): Nombre de la columna para calcular el IC.
        
    Returns:
        tuple: (límite_inferior, límite_superior) del intervalo de confianza.
    """
    # verificar que la columna exista
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el dataframe.")
       
    if metodo == "normal":
        # Calcular estadísticos
        media = df[columna].mean()
        desviacion_estandar = df[columna].std()
        n = len(df[columna])
        
        # definir % IC
        ci = 1 - (alpha) / 2
        z = norm.ppf(ci) 
        
        # limites del intervalo
        margen_error = z * (desviacion_estandar / np.sqrt(n))
        limite_inferior = round(float(media - margen_error),2)
        limite_superior = round(float(media + margen_error),2)

    
    elif metodo == "bootstrap":
        if seed is not None:
            np.random.seed(seed)
        
        # Generar muestras bootstrap
        bootstrap_means = []
        n = len(df[columna])
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(df[columna], size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        
        # Calcular percentiles para el IC
        limite_inferior = round(float(np.percentile(bootstrap_means, alpha / 2 * 100)), 2)
        limite_superior = round(float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100)), 2)
    
    else:
        raise ValueError("El método debe ser 'normal' o 'bootstrap'.")
    
    print(f"El RMSE del test se encuentra con un {100 * (1 - alpha):.1f}% de confianza entre {limite_inferior} y {limite_superior}")
    
    return limite_inferior, limite_superior

def select_best_features(X_train, y_train, score, cross_val, model="linear_regression", params=None, method="grid"):
    mlflow.autolog(disable=True)
    if model == "decision_tree":
        if method == "grid":
            search = GridSearchCV(
                estimator=DecisionTreeRegressor(),
                param_grid= (params or {}),
                scoring=score,
                cv=cross_val,
                n_jobs=-1
            )
        elif method == "bayes": 
            search = BayesSearchCV(
                estimator=DecisionTreeRegressor(),
                search_spaces= (params or {}),
                scoring=score,
                cv=cross_val,
                n_jobs=-1,
                random_state=seed
            )
        else: 
            raise ValueError("The method introduced is not valid. It must be either 'grid' or 'bayes'.")
        
        search.fit(X_train, y_train)

        # Get the best tree
        model = search.best_estimator_

    elif model == "linear_regression":
        model = LinearRegression()
        
    else:
        raise ValueError("The model introduced is not valid. It must be either 'linear_regression' or 'decision_tree'")

    rfecv = RFECV(estimator=model, cv=cross_val, scoring=score, n_jobs=-1)

    rfecv.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv.support_]

    print(f"Selected Features with RFECV: {selected_features}")

    return selected_features

def run_pipelines(X_train, y_train, preprocessing_pipeline, models, cross_val, score, verbosity, search_method="grid"):
    best_pipelines = []
    for model_name, (model, params) in models.items():
        pipeline = Pipeline(preprocessing_pipeline.steps + [('regressor', model)])

        
        if search_method == "grid":
            result = run_gridsearch_experiment(X_train=X_train, y_train=y_train, model_name=model_name, model=pipeline, 
                                               param_grid=params, cross_val=cross_val, score=score, verbosity=verbosity)
        elif search_method == "bayes":
            result = run_bayessearch_experiment(X_train=X_train, y_train=y_train, model_name=model_name, n_iter=30,
                                                model=pipeline, param_space=params, cross_val=cross_val, score=score, verbosity=verbosity)
        else: 
            raise ValueError("The search_method introduced is not valid")
        
        best_pipelines.append(result)

    best_model = max(best_pipelines, key=lambda x: x["score"])

    if verbosity:
        print("Best Model Overall:")
        print(f"Model Name: {best_model['model_name']}")
        print(f"Best Score: {best_model['score']}")
        print(f"Best Parameters: {best_model['params']}")

    return best_model
