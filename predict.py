import argparse
import json
import datetime

from args import get_parser, str2bool
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor

if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--model_id", type=str, default="-1",
                        help="ID (datetime) of pretrained model to use, or -1, -2, etc. to use last, previous from last, etc. model.")
    parser.add_argument("--load_scores", type=str2bool, default=False, help="To use already computed anomaly scores")
    parser.add_argument("--save_output", type=str2bool, default=False)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset

    # If no model_id is given, the last trained model is used
    if args.model_id is None:
        model_id = "-1"
    else:
        model_id = args.model_id

    if model_id.startswith('-'):
        dir_path = f"./output/{dataset}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        date_times = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
        date_times.sort()
        model_datetime = date_times[int(self.model_id)]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    model_path = f"./output/{dataset}/{model_id}"

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.window_size

    # Check that model is trained on specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    # Draw parameters
    window_size = model_args.window_size
    normalize = model_args.normalize
    n_epochs = model_args.epochs
    batch_size = model_args.bs
    init_lr = model_args.init_lr
    val_split = model_args.val_split
    shuffle_dataset = model_args.shuffle_dataset
    use_cuda = model_args.use_cuda
    print_every = model_args.print_every
    group_index = model_args.group[0]
    index = model_args.group[2:]
    args_summary = str(model_args.__dict__)

    # --------------------------- START EVALUATION -----------------------------
    # Get data from the dataset
    (x_train, _), (x_test, y_test) = get_data(args.dataset, normalize=normalize)

    # Cast data into tensor objects
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    # We want to perform forecasting/reconstruction on all features
    out_dim = n_features

    # Construct datasets from tensor objects
    train_dataset = SlidingWindowDataset(x_train, window_size)
    test_dataset = SlidingWindowDataset(x_test, window_size)

    # Create the data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    # Initialize the model
    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    # Load the model from the model path (parameters)
    # Now it has to be explicitly loaded, because it is not loaded using the Trainer class
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)

    # Setup the arguments for the prediction
    prediction_args = {
        'dataset': dataset,
        'scale_scores': args.scale_scores,
        "level": args.level,
        "q": args.q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": args.reg_level,
        "save_path": f"{model_path}",
    }

    # Creating a new summary-file each time a new prediction is made with a pre-trained model
    count = 0
    for filename in os.listdir(model_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"

    # Initialize a Predictor instance
    predictor = Predictor(model,
                          window_size,
                          n_features,
                          prediction_args,
                          summary_file_name=summary_file_name)

    # Start predictions
    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label,
                                load_scores=args.load_scores,
                                save_output=args.save_output)
