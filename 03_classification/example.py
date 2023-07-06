# 参考文献
# xgboostのパラメータ: https://xgboost.readthedocs.io/en/stable/parameter.html
# mlflowのxgboost.autolog: https://www.mlflow.org/docs/latest/python_api/mlflow.xgboost.html
# hydraのアウトプットディレクトリ: https://hydra.cc/docs/configure_hydra/workdir/
# hydraのカレントディレクトリ: https://hydra.cc/docs/upgrades/1.1_to_1.2/changes_to_job_working_dir/

import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

import warnings
#shapの警告を非表示にする
#NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator.
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import shap



#データの読み込み
#データの前処理
#モデルの学習
#モデルの推論

def create_new_run_id(cfg: DictConfig):
    client = mlflow.MlflowClient()
    
    #cfgからexperiment_nameを取得
    # experiment_name = cfg["experiment_name"]
    exp = client.get_experiment_by_name(cfg["experiment_info"]["name"])
    if exp is None:
        exp_id = client.create_experiment(cfg["experiment_info"]["name"])
    else:
        exp_id = exp.experiment_id
    
    run = client.create_run(exp_id)
    run_id = run.info.run_id
    return run_id

def eval_metrics(actual, pred):
    """一連の評価指標を計算する"""
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"ROC AUC: {roc_auc}")
    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "auc":roc_auc}

def load_data(
    loader, 
    **kwargs
    )->tuple:
    X,y = loader(**kwargs)
    # X_display,y_display = load_breast_cancer(return_X_y=True,as_frame=True)
    return X,y

def data_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: int = 0.2, 
    seed: int = 42,
    split_three_subset: bool = True,
    validation_size: int = 0.2
    )->tuple:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size= test_size, 
        random_state= seed, 
        stratify=y)
    
    if split_three_subset:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train, 
            test_size=0.2, 
            random_state= seed, 
            stratify=y_train
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test

def train(
    cfg: DictConfig,
    train_dataset: dict,
    val_dataset: dict = None,
    run_id: str = None,
):
    
    """モデルの学習"""    

    # print(cfg)
    # print(OmegaConf.to_yaml(cfg))
    params = OmegaConf.to_container(cfg["model"]["params"])
    tags = OmegaConf.to_container(cfg["model"]["tags"])
    mlflow.xgboost.autolog()    
    
    dtrain = xgb.DMatrix(train_dataset["X"], label=train_dataset["y"])
    if val_dataset is not None:
        dval = xgb.DMatrix(val_dataset["X"], label=val_dataset["y"])
        evals = [(dtrain, 'train'), (dval, 'eval')]
    else:
        evals = [(dtrain, 'train')]


    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags(tags)
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals = evals,
            early_stopping_rounds=100,
            )
        
    return model

def inference(
    test_dataset: dict,
    model: xgb.Booster,
    run_id: str = None,
):  
    dtest = xgb.DMatrix(test_dataset["X"], label=test_dataset["y"])
    best_iteration = model.best_iteration
    y_proba = model.predict(dtest,iteration_range=(0,best_iteration+1))
    y_predict = (y_proba > 0.5).astype(int)
    
    results = eval_metrics(test_dataset["y"], y_predict)
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(results)

    return y_predict, results
        
def mlflow_log_hydra(
    run_id: str ,
    artifact_path: str = None
    ):
    """hydraの設定をmlflowにログする"""
    # hydraの設定をmlflowにログする
    print(os.getcwd())
    output_hydra_dir = HydraConfig.get().runtime.output_dir
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(os.path.join(output_hydra_dir,".hydra"))
        
def mlflow_log_shap(
    run_id: str ,
    artifact_path: str = None
    ):
    """shapの設定をmlflowにログする"""
    # shapの設定をmlflowにログする
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(local_dir="./shap")


@hydra.main(version_base = None,config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(os.getcwd())
    print(HydraConfig.get().runtime.output_dir)
    output_hydra_dir = HydraConfig.get().runtime.output_dir
    
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    run_id = create_new_run_id(cfg)

    X,y = load_data(load_breast_cancer,return_X_y=True,as_frame=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(
        X,
        y,
        split_three_subset=True)
    
    
    train_dataset = {
        "X":X_train,
        "y":y_train
    }
    val_dataset = {
        "X":X_val,
        "y":y_val
    }
    test_dataset = {
        "X":X_test,
        "y":y_test
    }

    model = train(
        cfg = cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_id= run_id,
        )
    
    y_predict, results = inference(
        test_dataset=test_dataset,
        model=model,
        run_id=run_id
        )
    
    
    # #TODO
    # log hydra
    mlflow_log_hydra(run_id=run_id)
    # mlflow_log_shape(run_id=run_id)
    
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_output_dir = os.path.join(HydraConfig.get().runtime.output_dir,"shap")
    os.makedirs(shap_output_dir,exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar",show=False)
    plt.savefig(os.path.join(shap_output_dir,"shap_summary_bar.png"))
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="violin",show=False)
    plt.savefig(os.path.join(shap_output_dir,"shap_summary_violin.png"))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(shap_output_dir)


    
    # log shap
    
if __name__ == "__main__":
    main()