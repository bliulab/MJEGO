import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import time
import wandb
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# ==== WANDB ====
# export WANDB_BASE_URL=https://api.bandw.top
# export CUDA_VISIBLE_DEVICES=1
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["WANDB_API_KEY"] = "ed8dfbcd4ff447e018e6d6de7edb747352107329"
# os.environ["WANDB_MODE"] = "online"
# wandb sync wandb/offline-run-*

# ==== Import Module ====
import Utils.Constants as Constants
import Utils.params as params
import Utils.utils
from Dataset.Dataset import load_dataset # 数据集加载
from Models.idea222 import PFP as Model  # 模型定义

# ==== Parser Parameters ====
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=520, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--train_batch', type=int, default=50, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=50, help='Validation batch size.')
parser.add_argument('--seq', type=float, default=0.3, help='Sequence Identity (Sequence Identity).')
parser.add_argument("--ont", default='molecular_function', type=str, help='Ontology under consideration')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
print("Using device:", device)

# ==== Set DataLoad & Parameters ====
kwargs = {
    'seq_id': args.seq,
    'ont': args.ont,
    'session': 'train'
}

if args.ont == 'molecular_function':
    ont_kwargs = params.mol_kwargs
elif args.ont == 'cellular_component':
    ont_kwargs = params.cc_kwargs
elif args.ont == 'biological_process':
    ont_kwargs = params.bio_kwargs
else:
    raise ValueError(f"Unknown ontology type: {args.ont}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_class = len(Utils.utils.pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}'])

# ==== Comp Class Weights ====
def create_class_weights(cnter):
    class_weight_path = Constants.ROOT + "{}/{}/class_weights".format(kwargs['seq_id'], kwargs['ont'])
    if os.path.exists(class_weight_path + ".pickle"):
        print("Loading class weights")
        class_weights = Utils.utils.pickle_load(class_weight_path)
    else:
        print("Generating class weights")
        go_terms = Utils.utils.pickle_load(Constants.ROOT + "/go_terms")
        terms = go_terms['GO-terms-{}'.format(args.ont)]
        class_weights = [cnter[i] for i in terms]

    total = sum(class_weights)
    class_weights = torch.tensor([total / i for i in class_weights], dtype=torch.float).to(device)
    return class_weights

class_weights = create_class_weights(Utils.utils.class_distribution_counter(**kwargs))

# ==== Parser Dataset ====
dataset = load_dataset(root=Constants.ROOT, **kwargs)
labels = Utils.utils.pickle_load(Constants.ROOT + "label/" + "{}_labels".format(args.ont))

# edge_types = list(params.edge_types)

train_dataloader = DataLoader(dataset,
                              batch_size=args.train_batch,
                              drop_last=True,
                            #   exclude_keys=edge_types,
                              shuffle=True,
                              )

kwargs['session'] = 'validation'
val_dataset = load_dataset(root=Constants.ROOT, **kwargs)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=args.valid_batch,
                              drop_last=False,
                              shuffle=False,
                            #   exclude_keys=edge_types
                              )

print('========================================')
print(f'# training proteins: {len(dataset)}')
print(f'# validation proteins: {len(val_dataset)}')
print(f'# Number of classes: {num_class}')
print('========================================')

loaders = {'train': train_dataloader, 'valid': valid_dataloader}

# ==== Model, Optimizer, Loss ====
model = Model(**ont_kwargs).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=ont_kwargs['wd'])

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr,
    steps_per_epoch=len(train_dataloader),
    epochs=args.epochs, pct_start=0.1,
    anneal_strategy='cos', cycle_momentum=False)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

criterion = torch.nn.BCELoss(reduction='none')

# ==== Init ====
ckp_dir = "/data0/wengcchuang/model/Log/{}/{}/".format(args.ont, "model_test")
ckp_pth = ckp_dir + "current_checkpoint.pt"
current_epoch = 0
min_val_loss = np.Inf

if os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
    model, optimizer, current_epoch, min_val_loss = Utils.utils.load_ckp(ckp_pth, model, optimizer, device=device)
else:
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

print("Training model on epoch {}, with minimum validation loss as {}".format(current_epoch, min_val_loss))

# ==== Train Function ====
def train(start_epoch, min_val_loss, model, optimizer, criterion, data_loader):

    best_fmax = 0.0
    best_aupr = 0.0
    all_epoch_results = {}
    patience = 10  # Early stopping patience
    early_stop_counter = 0

    for epoch in range(start_epoch, args.epochs):
        print(f" ---------- Epoch {epoch} ----------")
        epoch_loss, val_loss = 0, 0
        t = time.time()

        lr_scheduler.step()
        model.train()
        y_true, y_preds = [], []

        for pos, data in enumerate(tqdm(data_loader['train'])):
            labs = [torch.tensor(labels[la], dtype=torch.float32).view(1, -1) for la in data['atoms'].protein]
            labs = torch.cat(labs, dim=0)
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, labs.to(device))
            loss = (loss * class_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip grad
            optimizer.step()
            epoch_loss += loss.item()

            y_true.append(labs.cpu())
            y_preds.append(output.cpu())

        y_true = torch.vstack(y_true).detach().numpy()
        y_preds = torch.vstack(y_preds).detach().numpy()
        train_F_max, train_AUPR = Utils.utils.evaluate_multilabel(y_true, y_preds)

        # Valid
        val_y_true, val_y_preds = [], []
        model.eval()
        with torch.no_grad():
            for data in tqdm(data_loader['valid']):
                labs = [torch.tensor(labels[la], dtype=torch.float32).view(1, -1) for la in data['atoms'].protein]
                labs = torch.cat(labs)
                output = model(data.to(device))

                _val_loss = criterion(output, labs.to(device))
                _val_loss = (_val_loss * class_weights).mean()
                val_loss += _val_loss.item()

                val_y_true.append(labs.cpu())
                val_y_preds.append(output.cpu())

        val_y_true = torch.vstack(val_y_true).detach().numpy()
        val_y_preds = torch.vstack(val_y_preds).detach().numpy()
        val_F_max, val_AUPR = Utils.utils.evaluate_multilabel(val_y_true, val_y_preds)

        all_epoch_results[f'epoch_{epoch}_y_true'] = val_y_true
        all_epoch_results[f'epoch_{epoch}_y_pred'] = val_y_preds

        print(f"Epoch: {epoch:04d}",
              f"train_loss: {epoch_loss:.4f}",
              f"train_fmax: {train_F_max:.4f}",
              f"train_aupr: {train_AUPR:.4f}",
              f"val_loss: {val_loss:.4f}",
              f"val_fmax: {val_F_max:.4f}",
              f"val_aupr: {val_AUPR:.4f}",
              f"time: {time.time() - t:.4f}s")
        
        # wandb.log({"train_loss": epoch_loss,
        #             "train_fmax": train_F_max,
        #             "train_aupr": train_AUPR,
        #             "val_loss": val_loss,
        #             "val_fmax": val_F_max,
        #             "val_aupr": val_AUPR,
        #             "time": time.time() - t})

        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        Utils.utils.save_ckp(checkpoint, False, ckp_pth, ckp_dir + "best_model.pt")

        if val_loss <= min_val_loss:
            print(f'Validation loss decreased ({min_val_loss:.6f} --> {val_loss:.6f}). Saving model ...')
            Utils.utils.save_ckp(checkpoint, True, ckp_pth, ckp_dir + "best_model.pt")
            min_val_loss = val_loss
            early_stop_counter = 0  # reset patience counter
        else:
            early_stop_counter += 1
            print(f'No improvement in val_loss for {early_stop_counter} epoch(s)')
            if early_stop_counter >= patience:
                print(f'Early stopping triggered! No improvement in {patience} epochs.')
                break

        if val_F_max > best_fmax:
            print(f'Best Fmax improved ({best_fmax:.4f} --> {val_F_max:.4f}). Saving model ...')
            Utils.utils.save_ckp(checkpoint, True, ckp_pth, ckp_dir + "best_fmax_model.pt")
            best_fmax = val_F_max

        if val_AUPR > best_aupr:
            print(f'Best AUPR improved ({best_aupr:.4f} --> {val_AUPR:.4f}). Saving model ...')
            Utils.utils.save_ckp(checkpoint, True, ckp_pth, ckp_dir + "best_aupr_model.pt")
            best_aupr = val_AUPR

    # === Save Result ===
    result_dir = f"/data0/wengcchuang/model/{args.ont}"
    os.makedirs(result_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(result_dir, "all_epochs_val_results.npz"),
        **all_epoch_results
    )

    return model

# ==== Set WANDB ====
# config = {
#     "learning_rate": args.lr,
#     "epochs": current_epoch,
#     "batch_size": args.train_batch,
#     "valid_size": args.valid_batch,
#     "weight_decay": ont_kwargs['wd']
# }
# wandb.init(project="Protein_Function_Prediction", config=config,
#            name="{}_{}".format("IDEA111", args.ont))

# ==== Training ====
trained_model = train(current_epoch, min_val_loss,
                      model=model, optimizer=optimizer,
                      criterion=criterion, data_loader=loaders)
