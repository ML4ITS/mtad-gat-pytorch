import json
from datetime import datetime
import torch.nn as nn

from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.window_size
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

    # Set output path for specific dataset
    output_path = f'output/{dataset}'
    # Setup a directory for logs in the output path
    log_dir = f'{output_path}/logs'

    # Make directories if they don't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get custom id for every run
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Setup an additional directory where the results of each different model
    # for the same dataset are stored
    save_path = f"{output_path}/{id}"

    # --------------------------- START TRAINING -----------------------------
    # Get data from the dataset
    (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)

    # Cast data into tensor objects
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    # We want to perform forecasting/reconstruction on all features
    out_dim = n_features
    print(f"Proceeding with forecasting and reconstruction of all {n_features} input features.")

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

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    # Set the criterion for each process: forecasting & reconstruction
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    # Initialize the Trainer module from training.py
    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    # Start training
    trainer.fit(train_loader, val_loader)
    # ---------------------------- END TRAINING ------------------------------

    # Plot training and validation losses
    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Perform evaluation on the test data
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Load the Trainer from the save_path
    # Thanks to the load() method, the model is loaded directly to the device
    # chosen in the initial arguments
    trainer.load(f"{save_path}/model.pt")

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
        "save_path": save_path,
    }

    # Get the model from the Trainer
    best_model = trainer.model

    # Initialize a Predictor instance
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    # Start predictions
    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label)

    # Workaround to write dimensions of dataset in config
    # to be used with the Plotter method
    args.__dict__['n_features'] = out_dim

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
