import pytorch_lightning as pl
from models import TypeRNet
from data import BigImagesDataModule
import config.config as configFile
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune, air
import datetime
import mlflow
from lightning.pytorch.loggers import MLFlowLogger


def train_typeR(config, num_gpus=0, num_workers=1):
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    num_gpus = num_gpus if num_gpus > 0 else 1

    mlflow = setup_mlflow(
        config,
        experiment_name=config.get("experiment_name", None),
        tracking_uri=config.get("tracking_uri", None),
    )

    model = TypeRNet(config)
    dm =BigImagesDataModule(
        imgs_dir = config["img_dir"],
        img_size = config["img_size"],
        img_size_test = config["img_size_test"],
        num_workers=num_workers,
        batch_size=config["batch_size"],
        val_ratio = config["val_ratio"],
        test_ratio = config["test_ratio"]
    )
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    mlflow.autolog()
    trainer = pl.Trainer(
        precision=config["precision"],
        max_epochs=config["num_epochs"],
        accelerator=accelerator,
        devices=num_gpus,
        deterministic=True,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
    )
    trainer.fit(model, dm)


def tune_typeR(
    num_samples=10,
    gpus_per_trial=0,
    num_workers=1,
):
    
    config = {
    "experiment_name": "simple Net "+ str(datetime.datetime.now()),
    "tracking_uri": str(configFile.ML_FLOW_TRACKING_URI),

    "lr" : tune.loguniform(1e-4, 1e-1),
    "alpha" : 1.,
    "beta" : 0.,
    "gamma" : 0.,

    "img_dir" : str(configFile.TRAINING_IMGS_DIR),
    "img_size" : 224,
    "img_size_test": 448,
    "batch_size": tune.choice([1]),
    "val_ratio" : 0.2,
    "test_ratio" : 0.2,
    "num_epochs": 5,
    "precision": tune.choice([32]),
    
    "font_path": str(configFile.FONT_PATH),
    "transposed_kernel_size" : 64,
    "transposed_stride": round(64*0.035),
    "transposed_padding": 31,
    "max_letter_per_pix": 5,
    "letters": configFile.TYPEWRITER_CONFIG["letterList"],
    "eps_out": 1./100,
    }

    # # Download data, done by pytorch
    # BigImagesDataModule(
    #     imgs_dir = config["img_dir"],
    #     img_size = config["img_size"],
    #     batch_size=4,
    #     val_ratio = config["val_ratio"],
    #     test_ratio = config["test_ratio"]
    # ).prepare_data()

    # Set the MLflow experiment, or create it if it does not exist.
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    # config = {
    #     "layer_1": tune.choice([32, 64, 128]),
    #     "layer_2": tune.choice([64, 128, 256]),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([32, 64, 128]),
    #     "experiment_name": experiment_name,
    #     "tracking_uri": mlflow.get_tracking_uri(),
    #     "data_dir": os.path.join(tempfile.gettempdir(), "mnist_data_"),
    #     "num_epochs": num_epochs,
    # }

    trainable = tune.with_parameters(
        train_typeR,
        num_gpus=gpus_per_trial,
        num_workers=num_workers,
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 2, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_typeR",
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)