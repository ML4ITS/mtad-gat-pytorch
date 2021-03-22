import argparse


def get_parsed_args():
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument("--dataset", type=str, default="smd")
    parser.add_argument(
        "--group",
        type=str,
        default="1-1",
        help="Required for smd dataset. <group_index>-<index>",
    )
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--target_col", type=int, default=None)

    # Model params
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    parser.add_argument("--autoenc_layers", type=int, default=1)
    parser.add_argument("--autoenc_hid_dim", type=int, default=128)
    parser.add_argument("--fc_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=150)

    # Train params
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--shuffle_dataset", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="models")
    parser.add_argument("--print_every", type=int, default=1)

    # Other
    parser.add_argument("--comment", type=str, default="")

    return parser.parse_args()
