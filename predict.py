import argparse
import json

from args import get_parser
from mtad_gat import MTAD_GAT
from prediction import Predictor
from utils import *

if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument(
        "--model_id", type=str, default=None, help="ID (datetime) of pretrained model to use, '-1' for latest"
    )
    args = parser.parse_args()
    print(args)

    if args.model_id is None:
        # Use latest model
        dir_path = f"./output/{args.dataset}/{args.group}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        subfolders.sort()
        model_id = subfolders[-1]

    else:
        model_id = args.model_id

    if args.dataset == "SMD":
        model_path = f"./output/{args.dataset}/{args.group}/{model_id}"
    elif args.dataset == "TELENOR":
        model_path = f"./output/{args.dataset}/{args.site}/{model_id}"
    else:
        model_path = f"./output/{args.dataset}/{model_id}"

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.lookback

    # Check that model is trained on specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    elif args.dataset == "TELENOR" and args.site != model_args.site:
        raise Warning(
            f"Model trained on Telenor site {model_args.site}, but asked to predict Telenor site {args.site}."
        )

    elif args.dataset == "SMD" and args.group != model_args.group:
        raise Warning(f"Model trained on SMD group {model_args.group}, but asked to predict SMD group {args.group}.")

    site = args.site
    do_preprocess = args.do_preprocess
    window_size = args.lookback
    horizon = args.horizon
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    test_split = args.test_size
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

    if args.dataset == "TELENOR":
        x_train, x_test = get_telenor_data(site, test_split=test_split, do_preprocess=do_preprocess)
        y_test = None
    elif args.dataset == "SMD":
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", do_preprocess=do_preprocess)
    else:
        (x_train, _), (x_test, y_test) = get_data(args.dataset, do_preprocess=do_preprocess)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    model = MTAD_GAT(
        n_features,
        window_size,
        model_args.horizon,
        out_dim,
        model_args.bs,
        kernel_size=model_args.kernel_size,
        dropout=model_args.dropout,
        gru_n_layers=model_args.gru_layers,
        gru_hid_dim=model_args.gru_hid_dim,
        autoenc_n_layers=model_args.autoenc_layers,
        autoenc_hid_dim=model_args.autoenc_hid_dim,
        forecast_n_layers=model_args.fc_layers,
        forecast_hid_dim=model_args.fc_hid_dim,
        use_cuda=model_args.use_cuda,
    )

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)

    level_dict = {"SMAP": 0.93, "MSL": 0.99, "SMD-1": 0.9950, "SMD-2": 0.9925, "SMD-3": 0.9999, "TELENOR": 0.99}
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level = level_dict[key]
    prediction_args = {
        "model_name": model_id,
        "target_dims": target_dims,
        "level": level,
        "q": args.q,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "save_path": model_path,
    }

    count = 0
    for filename in os.listdir(model_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"

    label = y_test[window_size:] if y_test is not None else None
    predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name="summary_1.txt")
    predictor.predict_anomalies(x_train, x_test, label, save_scores=False, load_scores=True, save_output=True)
