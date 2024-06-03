# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import re
from pathlib import Path

from ultralytics.utils import LOGGER, TESTS_RUNNING, colorstr

try:
    import mlflow

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(mlflow, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    mlflow = None


def on_pretrain_routine_end(trainer):
    """Logs training parameters to MLflow."""
    global mlflow, run, run_id, experiment_name

    if os.environ.get('MLFLOW_TRACKING_URI') is None:
        mlflow = None
    
    if mlflow:
        print(f'\n MLflow Start \n')
        mlflow_location = os.environ['MLFLOW_TRACKING_URI']  # "http://192.168.xxx.xxx:5000"
        print(f'\n Tracking URI: {mlflow_location} \n')
        mlflow.set_tracking_uri(mlflow_location)

        experiment_name = os.environ.get('MLFLOW_EXPERIMENT') or trainer.args.project or '/Shared/YOLOv8'
        print(f'\n Experiment Name: {experiment_name} \n')
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        prefix = colorstr('MLflow: ')
        try:
            run, active_run = mlflow, mlflow.active_run()
            if not active_run:
                print(f'\n MLflow Runing \n')
                name = os.environ.get('MLFLOW_RUN_NAME')
                active_run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=name)
            run_id = active_run.info.run_id
            LOGGER.info(f'{prefix}Using run_id({run_id}) at {mlflow_location}')
            run.log_params(vars(trainer.model.args))
        except Exception as err:
            LOGGER.error(f'{prefix}Failing init - {repr(err)}')
            LOGGER.warning(f'{prefix}Continuing without Mlflow')


def on_fit_epoch_end(trainer):
    """Logs training metrics to Mlflow."""
    if mlflow:
        metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
        run.log_metrics(metrics=metrics_dict, step=trainer.epoch)


def on_train_end(trainer):
    """Called at end of train loop to log model artifact info."""
    if mlflow:
        root_dir = trainer.save_dir 
        print(f'\n MLflow Tracking Directory: {root_dir} \n')
        run.log_artifact(trainer.last)
        run.log_artifact(trainer.best)
        run.log_artifact(trainer.save_dir)
        run.pyfunc.log_model(artifact_path=experiment_name,
                             code_path=[str(root_dir)],
                             artifacts={'model_path': f'{str(trainer.save_dir)}/weights/best.pt'},
                             python_model=run.pyfunc.PythonModel())
        mlflow.end_run()


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if mlflow else {}