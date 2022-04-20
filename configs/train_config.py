from .base_config import get_base_parser


def get_train_parser():
    parser = get_base_parser()

    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--save_ckpt", type=str, default=None)
    parser.add_argument("--save_log", type=str, default=None)
    parser.add_argument("--exclude_files", type=str, default="configs/excluded_files.txt")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--accelerator", default="auto")

    return parser
