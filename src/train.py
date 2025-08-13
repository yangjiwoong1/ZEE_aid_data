#!/usr/bin/env python

import sys
import os
import datetime
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import Compose
from src.datasets import load_aid_dataset
from src.lightning_modules import LitModel, ImagePredictionLogger
from src.modules import SRCNN
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

script_dir = os.path.dirname(os.path.realpath(__file__)) # 파일이 위치한 디렉토리 반환
repo_root = os.path.dirname(script_dir) # 디렉토리의 상위 디렉토리 반환
sys.path.append(repo_root) # 모듈 import 경로 추가

def cli_main():
    """
    CLI entrypoint for training.
    - Parses the CLI arguments
    - Intialises WandB
    - Loads the dataset
    - Generates the model
    - Runs the training
    - Finishes WandB logging
    """

    args = parse_arguments()
    set_random_seed(args)

    dataloaders = load_aid_dataset(**vars(args))
    generate_model_backbone(args, dataloaders)
    add_gpu_augmentations(args)

    model = generate_model(args)
    # Print model architecture (backbone)
    print("\n===== Model Backbone Architecture =====")
    print(model.backbone)
    print("======================================\n")

    add_callbacks(args, dataloaders)
    generate_and_run_trainer(args, dataloaders, model)
    finish_wandb_logging(args)


def generate_model(args):
    """ Generates a Lightning model from the arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.

    Returns
    -------
    LightningModule
        The Lightning model.
    """
    model = LitModel(**vars(args))  # Lightning model
    return model


def set_random_seed(args):
    """ Sets the random seed for reproducibility.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing the random seed (args.seed).
    """
    pl.seed_everything(args.seed)


def finish_wandb_logging(args):
    """ Finishes logging in WandB.
    Uploads the model to WandB if the upload_checkpoint flag is set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing the random seed (args.seed).
    """
    if args.use_wandb:
        if args.upload_checkpoint:
            best_model_path = args.checkpoint_callback.best_model_path
            wandb.save(best_model_path)

            # 마지막 모델도 저장하고 싶으면 
            # last_ckpt_path = os.path.join(args.checkpoint_callback.dirpath, "last.ckpt")
            # wandb.save(last_ckpt_path)
        wandb.finish()


def generate_and_run_trainer(args, dataloaders, model):
    """ Generates and runs the Lightning trainer.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    dataloaders : dict
        Dictionary containing the dataloaders for the training, validation and
        test set.
    model : LightningModule
        The Lightning model.
    """
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dataloaders["train"], dataloaders["val"])

    if not args.fast_dev_run:
        trainer.test(model=model, dataloaders=dataloaders["test"])


def add_callbacks(args, dataloaders):
    """ Adds logging and early stopping callbacks to the Lightning trainer.
    WandbLogger, ImagePredictionLogger and LearningRateMonitor are added.

    ModelCheckpoint is used to save the best model during training.
    EarlyStopping is used to stop training if the validation loss does not
    improve for 4 epochs.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    dataloaders : dict
        Dictionary containing the dataloaders for the training, validation and
        test set.
    """
    callbacks = []
    
    current_time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%m%d-%H:%M")

    # 가장 좋은 모델, 마지막 모델 저장
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/exp_{current_time}",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        verbose=True
    )

    callbacks.append(checkpoint_callback)

    if args.early_stop:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    # Print/Log a full model summary at the start of training
    callbacks.append(ModelSummary(max_depth=-1))
    
    # wandb가 활성화된 경우에만 wandb 관련 콜백 추가
    if args.use_wandb:
        tags = [] if not args.upload_checkpoint else ["inference"]

        vars(args)["logger"] = WandbLogger(
            project="zee_aid",
            name=f"{args.model}_{args.zoom_factor}x_{current_time}",
            tags=tags,
            config=args,
        )
        wandb.run.log_code("./src/")

        callbacks.extend([
            ImagePredictionLogger(
                train_dataloader=dataloaders["train"],
                val_dataloader=dataloaders["val"],
                test_dataloader=dataloaders["test"],
                log_every_n_epochs=1,
            ),
            LearningRateMonitor(logging_interval="step"),
        ])

    vars(args)["callbacks"] = callbacks
    vars(args)["checkpoint_callback"] = checkpoint_callback


def add_gpu_augmentations(args):
    """ Adds GPU augmentations to the Lightning model.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    """
    vars(args)["transform"] = Compose(
        [  # GPU augmentations
            # FilterData(values=(0,), thres=0.4, fill="zero", batched=True),
            # RandomRotateFlipDict(angles=[0, 90, 180, 270], batched=True),
        ]
    )


def generate_model_backbone(args, dataloaders):
    """
    Generates the model backbone from the arguments and the dataset.

    Args:
        args (argparse.Namespace): Arguments parsed from the CLI and model and project specific arguments.
        dataloaders (dict): Dictionary containing the dataloaders for the training, validation and
    """
    in_channels = dataloaders["train"].dataset[0]["lr"].shape[-3]
    out_channels = dataloaders["train"].dataset[0]["hr"].shape[-3]

    crop_ratio = args.chip_size[0] / args.output_size[0]
    output_height, output_width = args.output_size
    cropped_output_size = (
        round(output_height * crop_ratio),
        round(output_width * crop_ratio),
    )

    # TODO: pass hparams as kwargs
    model_cls = eval(args.model)  # Backbone
    vars(args)["backbone"] = model_cls(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=out_channels,
        kernel_size=args.kernel_size,
        residual_layers=args.residual_layers,
        output_size=cropped_output_size,
        zoom_factor=args.zoom_factor,
        sr_kernel_size=args.sr_kernel_size,
        padding_mode=args.padding_mode,
        use_dropout=args.use_dropout,
        use_batchnorm=args.use_batchnorm,
    )

def parse_arguments():
    """ Parses the arguments passed from the CLI using ArgumentParser and
    adds the model and project specific arguments.

    If a list of aois is passed, it is loaded and added to the args namespace.

    Returns
    -------
    argparse.Namespace
        Namespace containing the parsed arguments.
    """
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser) # gpus, precision, max-epochs 등
    parser = LitModel.add_model_specific_args(parser) # 모델 관련 인자 추가
    parser = add_project_specific_arguments(parser) # 프로젝트 관련 인자 추가
    args = parser.parse_args()
    return args


def add_project_specific_arguments(parser):
    """ Adds the project specific arguments to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to add the project specific arguments to.

    Returns
    -------
    argparse.ArgumentParser
        Argument parser with the project specific arguments added.
    """

    parser.add_argument(
        "--root", default="../AID-dataset/", type=str, help="Root folder of the dataset."
    )
    parser.add_argument(
        "--data_split_seed",
        default=42,
        type=int,
        help="Separate seed to ensure the train/val/test split remains the same.",
    )
    parser.add_argument(
        "--no_normalize_lr",
        action="store_false",
        help="Normalize the low resolution images.",
    )
    parser.add_argument(
        "--no_normalize_hr",
        action="store_false",
        help="Normalize the high resolution images.",
    )
    parser.add_argument(
        "--randomly_rotate_and_flip_images",
        action="store_true",
        help="Randomly rotate and flip the images.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset.",
    )

    # Training arguments
    parser.add_argument(
        "--seed", default=1337, type=int, help="Randomization seed."
    )  # random seed
    parser.add_argument("--num_workers", default=0, type=int)  # CPU cores
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--subset_train",
        default=1.0,
        type=float,
        help="Fraction of the training dataset.",
    )
    # Logging arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",  # 기본값 False
        help="Enable WandB logging"
    )
    parser.add_argument(
        "--benchmark_logging",
        action="store_true",
        help="Enable benchmark mode",
    )
    parser.add_argument(
        "--upload_checkpoint",
        action="store_true",
        help="Uploads the model checkpoint to WandB.",
    )

    # Early stopping arguments
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable early stopping.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum change in the monitored quantity to qualify as an improvement.",
    )

    return parser

if __name__ == "__main__":
    cli_main()
