import datetime
import os
from argparse import ArgumentParser
from pathlib import Path
import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from fane.data.downstream.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)

from fane.data.downstream.data_module import DataModule
from fane.data.downstream.transforms import DataTransforms, Moco2Transform

from fane.models.fane import Fane
from fane.downstream.ssl_finetuner import SSLFineTuner
from fane.utils.log_to_file import LogLossToFileCallback

from codes.fane.encoders.image.resnet_encoder import ResEncoder
from codes.fane.encoders.image.vit_encoder import VitEncoder
from codes.fane.encoders.language.bert import ClinicalBERT

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# torch.use_deterministic_algorithms(True, warn_only=True)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))



def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert", help='chexpert or rsna or covidx')
    parser.add_argument("--ckpt_path", type=str, default="xxx.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--base_model", type=str, default="resnet50", help="resnet50 or vit")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "chexpert":
        # define datamodule
        # check transform here
        datamodule = DataModule(CheXpertImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 5
        multilabel = True
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    if args.base_model == 'resnet50':
        image_encoder = ResEncoder(model_name = "resnet_50", weights_path = "xxxx.pth")
        text_encoder = ClinicalBERT()
        args.in_features = 2048

    else:
        image_encoder = VitEncoder(model_name='vit_base', output_dim=128, image_size=224, weights_path = "xxxxx.pth")
        text_encoder = ClinicalBERT()
        args.in_features = 768

    mgca = Mcga.load_from_checkpoint(args.ckpt_path,
                                     text_encoder=text_encoder,
                                     image_encoder=image_encoder,
                                     down_stream=True,
                                     image_encoder_mode = args.base_model,
                                     gpus=[0],
                                     strict=False)


    args.model_name = args.base_model
    args.backbone = mgca.image_encoder
    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # ckpt_dir = os.path.join(
    #     BASE_DIR, f"../../../data/ckpts/mgca_finetune/{extension}")

    ckpt_dir = os.path.join('/data', f"HAMeR/classification/{args.base_model}/{args.dataset}_{args.data_pct}_{extension}_full_loss_threshold_0.95")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger_dir = os.path.join('/data', f"HAMeR/classification/{args.base_model}/{args.dataset}_{args.data_pct}_{extension}_full_loss_threshold_0.95/logs")
    os.makedirs(logger_dir, exist_ok=True)

    file_path = Path(os.path.join('/data', f"HAMeR/classification/{args.base_model}/{args.dataset}_{args.data_pct}_{extension}_full_loss_threshold_0.95/training.log"))
    file_path.touch(exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min"),
        LogLossToFileCallback(log_file=file_path)
    ]

    tensor_logger = TensorBoardLogger(save_dir=logger_dir, name=f"HAMeR_{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        # deterministic=True,
        callbacks=callbacks,
        logger=tensor_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()





