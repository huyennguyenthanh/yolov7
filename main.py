import os
import logging
import argparse
import torch
from trainer.semi_trainer import SemiTrainer
from trainer.trainer import Trainer
from utils.general import set_logging

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def parse_opt():
    """Train Config

    Returns:
        Namespace(semi=False, weights='yolo7.pt', cfg='', data='data/coco.yaml', hyp='data/hyp.scratch.p5.yaml',
        epochs=300, batch_size=16, img_size=[640, 640], rect=False, resume=False, nosave=False, notest=False,
        noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='',
        multi_scale=False, single_cls=False, adam=False, sync_bn=False, local_rank=-1, workers=8,
        project='runs/train', entity=None, name='exp', exist_ok=False, quad=False,
        linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1,
        artifact_alias='latest', freeze=[0], v5_metric=False)
    """
    parser = argparse.ArgumentParser()
    # Type
    parser.add_argument("--semi", action="store_true", help="Semi Supervised Training")
    # Config
    parser.add_argument(
        "--weights", type=str, default="yolo7.pt", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--data", type=str, default="data/coco.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="data/hyp.scratch.p5.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train multi-class data as single-class",
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon"
    )
    parser.add_argument(
        "--upload_dataset",
        action="store_true",
        help="Upload dataset as W&B artifact table",
    )
    parser.add_argument(
        "--bbox_interval",
        type=int,
        default=-1,
        help="Set bounding-box image logging interval for W&B",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=-1,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument(
        "--artifact_alias",
        type=str,
        default="latest",
        help="version of dataset artifact to be used",
    )
    parser.add_argument(
        "--freeze",
        nargs="+",
        type=int,
        default=[0],
        help="Freeze layers: backbone of yolov7=50, first3=0 1 2",
    )
    parser.add_argument(
        "--v5-metric",
        action="store_true",
        help="assume maximum recall as 1.0 in AP calculation",
    )
    opt = parser.parse_args()

    return opt


def main(opt):

    # Set DDP variables
    opt.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    set_logging(opt.global_rank)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(opt)
    print(opt.cfg)
    print(opt.hyp)

    if opt.semi:
        trainer = SemiTrainer(opt, device)
    else:
        trainer = Trainer(opt, device)

    trainer.train()
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info("Destroying process group... ")
        # dist.destroy_process_group()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
