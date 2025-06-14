import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
NUM_ROUNDS = 6
from collections import defaultdict
from lightning.pytorch.loggers import WandbLogger
import datetime
from torchvision import transforms, models
import torch
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



def fix_seeds(seed=42):
    """Fixes random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ObjectDataset(Dataset):
    """
    Holds (gesture, object) pairs for training.
    """
    def __init__(self, X_pairs, y_labels, device='cpu'):
        """
        X_pairs: 2D numpy array of shape [num_samples, embedding_dim]
        y_labels: 1D numpy array of shape [num_samples]
        """
        self.X = torch.tensor(X_pairs, dtype=torch.float32).to(device)
        self.y = torch.tensor(y_labels, dtype=torch.float32).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiClassMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[100], num_classes=70):
        """
        input_dim: size of concatenated (gesture_emb, object_emb)
        hidden_dims: list of hidden layer sizes, e.g. [100, 50]
        """
        super(MultiClassMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: shape (batch_size, input_dim)
        returns: shape (batch_size, 1) (logits for binary classification)
        """
        return self.net(x)


def reference_classification_mlp_torch(
    embeddings_type,
    gestures_info,
    num_rounds=5,
    hidden_dims=[100],
    epochs=10,
    batch_size=32,
    lr=1e-3,
    device=None,
    pair='pairs',
    baseline=False
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embeddings type: {embeddings_type}")
    print(f"Training on {device}")
    le = LabelEncoder()
    le.fit(gestures_info['referent_clean'].unique())

    results_per_round = {}
    all_eval_pairs = []
    
    for round_idx in tqdm(range(num_rounds)):
        train_data = gestures_info[gestures_info['pair'] != pair]
        test_data  = gestures_info[(gestures_info['pair'] == pair) & (gestures_info['round'] == (round_idx + 1))]

        if not baseline:
            pair_train_data = gestures_info[(gestures_info['pair'] == pair) & (gestures_info['round'] < (round_idx + 1))]
            num_pair_gestures = pair_train_data.shape[0]
            # remove the same number of samples from the training data
            train_data = train_data.sample(n=train_data.shape[0] - num_pair_gestures)
            train_data = pd.concat([train_data, pair_train_data])

        if not baseline or (baseline and round_idx == 0):

            X_train = train_data[embeddings_type].to_numpy()
            X_train = np.stack(X_train, axis=0).astype(np.float32).squeeze()
            y_train = le.transform(train_data['referent_clean'])
            num_classes = len(le.classes_)
    
            train_dataset = ObjectDataset(X_train, y_train, device=device)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        
            input_dim = X_train.shape[1]  # dimension of gesture_emb + object_emb
            model = MultiClassMLP(input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_y = batch_y.long()
                    optimizer.zero_grad()
                    logits = model(batch_X).squeeze(-1) 
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * batch_X.size(0)

                epoch_loss = total_loss / len(train_dataset)
                print(f"Round {round_idx+1}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        # store statistics related to the distribution of the positive and negative pairs
        model.eval()
        predictions = []
        references  = []
        
        with torch.no_grad():
            X_test = test_data[embeddings_type].to_numpy()
            for i in range(len(X_test)):
                X_test[i] = np.array(X_test[i]).squeeze()
            if test_data.shape[0] == 0:
                continue
            elif len(X_test) == 1:
                X_test = np.stack(X_test, axis=0).astype(np.float32)
            else:
                X_test = np.stack(X_test, axis=0).astype(np.float32).squeeze()
            y_test = le.transform(test_data['referent_clean'])
            test_dataset = ObjectDataset(X_test, y_test, device=device)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            print(f"Testing on {len(test_dataset)} samples...")
            if len(test_dataset) == 0:
                continue
            for batch_X, batch_y in test_loader:
                batch_y = batch_y.long()
                logits = model(batch_X).squeeze(-1)
                _, preds = torch.max(logits, 1)
                predictions.extend(preds.cpu().numpy())
                references.extend(batch_y.cpu().numpy())
                for test_idx, test_sample in enumerate(batch_y):
                    pairs_infon = {
                        'Referent': le.inverse_transform([test_sample.cpu().numpy()])[0],
                        'Logit': logits[test_idx].cpu().numpy(),
                        'Predicted Object': le.inverse_transform([preds[test_idx].cpu().numpy()])[0],
                        'Round': round_idx + 1,
                        'Embedding Type': embeddings_type,
                        'Correct': test_sample == preds[test_idx],
                        # 'Pair_Speaker': test_data['speaker'].iloc[test_idx],
                    }
                all_eval_pairs.append(pairs_infon)

        
        y_true_enc = np.array(references)
        y_pred_enc = np.array(predictions)
        macro_precision = precision_score(y_true_enc, y_pred_enc, average='macro')
        macro_recall    = recall_score(y_true_enc, y_pred_enc, average='macro')
        macro_f1        = f1_score(y_true_enc, y_pred_enc, average='macro')
        micro_precision = precision_score(y_true_enc, y_pred_enc, average='micro')
        micro_recall    = recall_score(y_true_enc, y_pred_enc, average='micro')
        micro_f1        = f1_score(y_true_enc, y_pred_enc, average='micro')
        accuracy        = accuracy_score(y_true_enc, y_pred_enc)

        results_per_round[f'Round {round_idx + 1}'] = {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'accuracy': accuracy
        }

    return results_per_round, all_eval_pairs


if __name__ == '__main__':
    fix_seeds(seed=42)

    gestures_info_exploded = pd.read_pickle('data/gestures_info_exploded_subparts.pkl')
    
    embeddings_types = [
        ('multimodal-x-skeleton-semantic', 'Multimodal-X'),
        ('unimodal_skeleton', 'Unimodal'),
    ]

    param_grid = {
        'batch_size': [32],
        'learning_rate': [0.0001],
        'hidden_dims': [[300, 150]],
        'epochs': [100]
    }
    results_list = []
    results_stats = []
    for baseline in [False]:
        pairs = gestures_info_exploded['pair'].unique()
        for batch_size, learning_rate, hidden_dims, epochs, (emb_type, emb_name) in product(
            param_grid['batch_size'],
            param_grid['learning_rate'],
            param_grid['hidden_dims'],
            param_grid['epochs'],
            embeddings_types
        ):
            for pair in pairs:
                print(f"Running for {emb_name} embeddings using MLP with hidden_dims={hidden_dims}, "
                    f"batch_size={batch_size}, lr={learning_rate}... Pair: {pair}")
                ref_res_results, all_eval_pairs = reference_classification_mlp_torch(
                    embeddings_type=emb_type,
                    gestures_info=gestures_info_exploded,
                    num_rounds=NUM_ROUNDS,
                    hidden_dims=hidden_dims,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=learning_rate,
                    pair=pair,
                    baseline=baseline
                )
                results_stats.extend(all_eval_pairs)
                if True:
                    for round_name, metrics in ref_res_results.items():
                        result_dict = {
                            'Embedding Type': emb_name,
                            'Round': round_name,
                            'Task': 'Reference Resolution',
                            'Hidden Layer Sizes': str(hidden_dims),
                            'Epochs': epochs,
                            'Batch Size': batch_size,
                            'Learning Rate': learning_rate,
                            'pair': pair,
                            'Baseline': baseline
                        }
                        result_dict.update(metrics)  # Merge metrics into the dictionary
                        results_list.append(result_dict)
                    if True:
                        # Save results to CSV after each run
                        results_to_save = pd.DataFrame(results_list)
                        # check if file exists
                        results_path = 'results/reference_resolution_with_dialogue_history_results.csv'
                        try:
                            df = pd.read_csv(results_path)
                            df = pd.concat([df, results_to_save])
                            df.to_csv(results_path, index=False)
                        except FileNotFoundError:
                            results_to_save.to_csv(results_path, index=False)
                        results_list = []
    df_stats = pd.DataFrame(results_stats)
    df_stats.to_csv('results/reference_resolution_with_dialogue_history_results_stats.csv', index=False)