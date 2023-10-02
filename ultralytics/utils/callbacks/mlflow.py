# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
MLflow Logging for Ultralytics YOLO.

<<<<<<< HEAD
This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.

Commands:
    1. To set a project name:
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

    2. To set a run name:
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

    3. To start a local MLflow server:
        mlflow server --backend-store-uri runs/mlflow
       It will by default start a local server at http://127.0.0.1:5000.
       To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.

    4. To kill all running MLflow server instances:
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr

try:
    import os

    assert not TESTS_RUNNING or "test_mlflow" in os.environ.get("PYTEST_CURRENT_TEST", "")  # do not log pytest
    assert SETTINGS["mlflow"] is True  # verify integration is enabled
    import mlflow

    assert hasattr(mlflow, "__version__")  # verify package is not directory
    from pathlib import Path

    PREFIX = colorstr("MLflow: ")
    SANITIZE = lambda x: {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}

=======
import os
import re
from pathlib import Path

from ultralytics.utils import LOGGER, TESTS_RUNNING, colorstr

try:
    import mlflow

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(mlflow, '__version__')  # verify package is not directory
>>>>>>> a2f3f15d (modify MLflow)
except (ImportError, AssertionError):
    mlflow = None


def on_pretrain_routine_end(trainer):
<<<<<<< HEAD
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
    from the trainer.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

    Global:
        mlflow: The imported mlflow module to use for logging.

    Environment Variables:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
        MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after the end of training.
    """
    global mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/YOLOv8"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n" f"{PREFIX}WARNING ⚠️ Not tracking this run")


def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(
            metrics={
                **SANITIZE(trainer.lr),
                **SANITIZE(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
            step=trainer.epoch,
        )
=======
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
>>>>>>> a2f3f15d (modify MLflow)


def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=SANITIZE(trainer.metrics), step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if mlflow:
<<<<<<< HEAD
        mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
        for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
            if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
                mlflow.log_artifact(str(f))
        keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
        if keep_run_active:
            LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
        else:
            mlflow.end_run()
            LOGGER.debug(f"{PREFIX}mlflow run ended")

        LOGGER.info(
            f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n"
            f"{PREFIX}disable with 'yolo settings mlflow=False'"
        )
=======
        # root_dir = Path(__file__).resolve().parents[3]
        root_dir = Path('/home/mmaaou/PESLAU/runs/').resolve()
        print(f'\n MLflow Tracking Directory: {root_dir} \n')
        run.log_artifact(trainer.last)
        run.log_artifact(trainer.best)
        run.pyfunc.log_model(artifact_path=experiment_name,
                             code_path=[str(root_dir)],
                             artifacts={'model_path': str(trainer.save_dir)},
                             python_model=run.pyfunc.PythonModel())
        # mlflow.end_run()
>>>>>>> a2f3f15d (modify MLflow)


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}
)
