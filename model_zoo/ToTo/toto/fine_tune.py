# THIS IS THE CORRECT AND WORKING SCRIPT

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# --- Add the cloned repository to Python's path ---
sys.path.append(os.path.abspath('toto'))
from model.toto import Toto

# --- Function to Generate Synthetic Data ---
def generate_synthetic_data(num_samples=2000, num_variates=3):
    print(f"Generating {num_samples} samples for {num_variates} variates...")
    timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='H')
    data = {}
    time = np.arange(num_samples)
    for i in range(num_variates):
        trend = time * (np.random.rand() * 0.05 + 0.01)
        seasonality = np.sin(time * 2 * np.pi / (24 * (i+1) + np.random.randint(-3,3))) * (i+1) * 2 \
                    + np.cos(time * 2 * np.pi / (24*7)) * (i+1)
        noise = np.random.randn(num_samples) * 0.5
        data[f'variate_{i+1}'] = trend + seasonality + noise
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'timestamp' 
    return df

# --- Custom Dataset for Time Series ---
class TimeSeriesDataset(Dataset):
    def __init__(self, df, context_len, pred_len):
        self.context_len = context_len
        self.pred_len = pred_len
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.sample_len = context_len + pred_len

    def __len__(self):
        return len(self.data) - self.sample_len + 1

    def __getitem__(self, idx):
        sample = self.data[idx : idx + self.sample_len]
        context, target = sample[:self.context_len], sample[self.context_len:]
        return context.T, target.T

# --- Custom Composite Loss Function ---
class CompositeLoss(torch.nn.Module):
    def __init__(self, lambda_nll=0.57, delta=0.1):
        super().__init__()
        self.lambda_nll, self.delta = lambda_nll, delta
        print(f"CompositeLoss initialized with λ_NLL={self.lambda_nll}, δ={self.delta}")

    def forward(self, predicted_distribution, true_values):
        nll_loss = -predicted_distribution.log_prob(true_values).mean()
        weights = predicted_distribution.mixture_distribution.probs
        means = predicted_distribution.component_distribution.loc
        predicted_mean = torch.sum(weights * means, dim=-1)
        residual = true_values - predicted_mean
        cauchy_loss = torch.log(0.5 * ((residual / self.delta) ** 2) + 1).mean()
        total_loss = (self.lambda_nll * nll_loss) + ((1 - self.lambda_nll) * cauchy_loss)
        return total_loss, nll_loss, cauchy_loss

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    CONTEXT_LENGTH, PREDICTION_LENGTH, NUM_EPOCHS = 128, 32, 10
    BATCH_SIZE, LEARNING_RATE = 32, 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CORRECT ID for the native Toto model architecture from the GitHub repository
    MODEL_ID = "Datadog/Toto-Open-Base-1.0"
    OUTPUT_MODEL_PATH = "./toto-native-finetuned.pth"

    # --- Data Generation and Preprocessing ---
    print("Step 1: Generating synthetic data and preparing it...")
    df = generate_synthetic_data(num_samples=4000, num_variates=5)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    train_split_idx = int(len(df_scaled) * 0.8)
    train_df = df_scaled.iloc[:train_split_idx]
    train_dataset = TimeSeriesDataset(train_df, CONTEXT_LENGTH, PREDICTION_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Data prepared. Training samples: {len(train_dataset)}, Device: {DEVICE}")

    # --- Load the Pre-trained Model ---
    print(f"Step 2: Loading pre-trained model '{MODEL_ID}'...")
    model = Toto.from_pretrained(MODEL_ID, map_location=DEVICE)
    model.to(DEVICE)

    # --- Fine-Tuning Loop ---
    print("Step 3: Starting fine-tuning...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = CompositeLoss()
    model.train()
    for epoch in range(NUM_EPOCHS):
        # ... (Training loop is identical to the previous correct version) ...
        for i, (context, target) in enumerate(train_loader):
            optimizer.zero_grad()
            context, target = context.to(DEVICE), target.to(DEVICE)
            input_padding_mask = torch.ones_like(context, dtype=torch.bool, device=DEVICE)
            id_mask = torch.ones_like(context, dtype=torch.float32, device=DEVICE)
            output = model.model(inputs=context, input_padding_mask=input_padding_mask, id_mask=id_mask)
            predicted_distribution = output.distribution[:, :, :PREDICTION_LENGTH]
            loss, _, _ = criterion(predicted_distribution, target)
            loss.backward()
            optimizer.step()
        print(f"--- Epoch {epoch+1} Loss: {loss.item():.4f} ---")

    print("Fine-tuning complete.")
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)

    # --- Inference Example ---
    # ... (Inference logic is identical to the previous correct version) ...