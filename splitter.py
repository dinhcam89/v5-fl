import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ==== Cấu hình trực tiếp trong file ====
NUM_CLIENTS = 9
SEED = 42
DATA_PATH = "data"  # Thư mục chứa creditcard.csv
VERBOSE = True
# =======================================

def split_dataset_iid(data: pd.DataFrame, num_clients: int = NUM_CLIENTS, seed: int = SEED):
    """Chia data theo IID"""
    data_shuffled = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    client_data = np.array_split(data_shuffled, num_clients)
    return client_data

def split_dataset_noniid_label_skew(data: pd.DataFrame, label_col: str = "Class", num_clients: int = NUM_CLIENTS, seed: int = SEED):
    """
    Non-IID: chia theo label-skew, mỗi client chủ yếu thấy một số nhãn.
    Với fraud detection (binary), sẽ chia lệch nhãn 0/1 giữa các client.
    """
    np.random.seed(seed)
    clients_data = [[] for _ in range(num_clients)]
    classes = data[label_col].unique()

    for c in classes:
        class_data = data[data[label_col] == c]
        parts = np.array_split(class_data, num_clients)
        for i in range(num_clients):
            if c == 1:  # thiểu số
                if i % 2 == 0:
                    clients_data[i].append(parts[i])
            else:
                clients_data[i].append(parts[i])

    return [pd.concat(client_parts).sample(frac=1, random_state=seed) for client_parts in clients_data]

def save_client_datasets(clients_data, output_dir="data/clients"):
    os.makedirs(output_dir, exist_ok=True)
    for i, df in enumerate(clients_data):
        df.to_csv(os.path.join(output_dir, f"client_{i}.csv"), index=False)
        if VERBOSE:
            print(f"✅ Saved client {i} data with {len(df)} samples")

# ================== CÁCH DÙNG ==================

# Load dataset
df = pd.read_csv(f"{DATA_PATH}/creditcard.csv")

# Chọn 1 trong 2 phương pháp chia:
clients_data = split_dataset_iid(df)  # IID
#clients_data = split_dataset_noniid_label_skew(df, label_col="Class")  # non-IID

# Lưu thành từng file
save_client_datasets(clients_data)
