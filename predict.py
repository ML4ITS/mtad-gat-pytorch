import argparse
import json

from args import get_parser
from mtad_gat import MTAD_GAT
from prediction import Predictor
from utils import *

if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--model", type=str, required=True, help="Name of model to use")
    args = parser.parse_args()
    print(args)

    model = args.model

    # Peak-Over-Threshold args
    # Recommend values for `level`:
    # SMAP: 0.93
    # MSL: 0.99
    # SMD group 1: 0.9950
    # SMD group 2: 0.9925
    # SMD group 3: 0.9999

    if args.level is not None:
        level = args.level
    else:
        level_dict = {
            "SMAP": 0.93,
            "MSL": 0.99,
            "SMD-1": 0.9950,
            "SMD-2": 0.9925,
            "SMD-3": 0.9999,
            "TELENOR": 0.99
        }
        key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
        level = level_dict[key]

    pre_trained_model_path = f"models/{model}/{model}"
    # Check that model exist
    if not os.path.isfile(f"{pre_trained_model_path}_model.pt"):
        raise Exception(f"Model <{pre_trained_model_path}_model.pt> does not exist.")

    # Get configs of model
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{pre_trained_model_path}_config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.lookback

    # Check that model is trained on specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    elif args.dataset == "TELENOR" and args.site != model_args.site:
        raise Warning(f"Model trained on Telenor site {model_args.site}, but asked to predict Telenor site {args.site}.")

    elif args.dataset == "SMD" and args.group != model_args.group:
        raise Warning(f"Model trained on SMD group {model_args.group}, but asked to predict SMD group {args.group}.")

    if args.dataset == "TELENOR":
        output_path = f"output/TELENOR/{args.site}"

    elif args.dataset == "SMD":
        output_path = f"output/SMD/{args.group}"
    else:
        output_path = f"output/{args.dataset}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.dataset == "TELENOR":
        x_train, x_test = get_telenor_data(args.site, test_split=0.1, do_preprocess=args.do_preprocess)
    elif args.dataset == "SMD":
        group_index = args.group[0]
        index = args.group[2:]
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}")
    else:
        (x_train, _), (x_test, y_test) = get_data(args.dataset)

    save_scores = args.save_scores
    load_scores = args.load_scores

    label = y_test[window_size:] if y_test is not None else None
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
    load(model, f"{pre_trained_model_path}_model.pt", device=device)
    model.to(device)

    prediction_args = {
        'model_name': args.model,
        'target_dims': target_dims,
        'level': level,
        'q': args.q,
        'use_mov_av': args.use_mov_av,
        'gamma': args.gamma,
        'save_path': output_path
    }
    predictor = Predictor(
        model,
        window_size,
        n_features,
        prediction_args
    )
    predictor.predict_anomalies(x_train, x_test, label, save_scores=save_scores, load_scores=load_scores)
