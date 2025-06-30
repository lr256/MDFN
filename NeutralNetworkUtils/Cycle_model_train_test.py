import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from termcolor import colored
import heapq

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = 5
    batch_size = 96*2
    lr = 0.00001
    epochs = 300
    lambda_cycle = 10
    pos_epsilon = 0  #
    input_dim_A = 5
    input_dim_B = 1
    group_sizes = [1, 1, 1, 1, 1]

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_A2B = Generator(Config.input_dim_A, Config.input_dim_B).to(Config.device)
        self.G_B2A = Generator(Config.input_dim_B, Config.input_dim_A).to(Config.device)
        self.D_A = Discriminator(Config.input_dim_A).to(Config.device)
        self.D_B = Discriminator(Config.input_dim_B).to(Config.device)
        
        self.optimizer_G = optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=Config.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=Config.lr, betas=(0.5, 0.999)
        )
        
        self.criterion_gan = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()

    def train_step(self, real_A, real_B):
        # 训练生成器
        self.optimizer_G.zero_grad()
        
        # 前向传播
        fake_B = self.G_A2B(real_A)
        cycle_A = self.G_B2A(fake_B)
        
        fake_A = self.G_B2A(real_B)
        cycle_B = self.G_A2B(fake_A)
        
        # 对抗损失
        pred_fake_B = self.D_B(fake_B)
        loss_G_A2B = self.criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B))
        
        pred_fake_A = self.D_A(fake_A)
        loss_G_B2A = self.criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A))
        
        # 循环一致性损失
        loss_cycle_A = self.criterion_cycle(cycle_A, real_A)
        loss_cycle_B = self.criterion_cycle(cycle_B, real_B)
        
        # 总损失
        total_loss_G = loss_G_A2B + loss_G_B2A + Config.lambda_cycle*(loss_cycle_A + loss_cycle_B)
        total_loss_G.backward()
        self.optimizer_G.step()
        
        # 训练判别器
        self.optimizer_D.zero_grad()
        
        # 真实数据
        pred_real_A = self.D_A(real_A)
        loss_D_real_A = self.criterion_gan(pred_real_A, torch.ones_like(pred_real_A))
        
        pred_real_B = self.D_B(real_B)
        loss_D_real_B = self.criterion_gan(pred_real_B, torch.ones_like(pred_real_B))
        
        # 生成数据
        pred_fake_A = self.D_A(fake_A.detach())
        loss_D_fake_A = self.criterion_gan(pred_fake_A, torch.zeros_like(pred_fake_A))
        
        pred_fake_B = self.D_B(fake_B.detach())
        loss_D_fake_B = self.criterion_gan(pred_fake_B, torch.zeros_like(pred_fake_B))
        
        # 总判别器损失
        total_loss_D =(loss_D_real_A + loss_D_fake_A + loss_D_real_B + loss_D_fake_B) / 2
        total_loss_D.backward()
        self.optimizer_D.step()
        
        return {
            "G_loss": total_loss_G.item(),
            "D_loss": total_loss_D.item(),
            "cycle_loss":(loss_cycle_A.item() + loss_cycle_B.item())/2# (loss_cycle_A.item() + loss_cycle_B.item())/2
        }


class ClinicalDataset(Dataset):
    def __init__(self, data_df, is_train=True):
        self.is_train = is_train
        self.group_indices = []
        start = 0
        for size in Config.group_sizes:
            self.group_indices.append((start, start+size))
            start += size
        if is_train:
            self.features = data_df.iloc[:, 0:5].values.astype(np.float32)
            self.labels = data_df.iloc[:, -1:].values.astype(np.float32)
        else:
            self.data = data_df.values.astype(np.float32)

    def __len__(self):
        return len(self.features) if self.is_train else len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            domain_A =  np.clip(self.features[idx], Config.pos_epsilon, None)
            domain_B =  np.clip(self.labels[idx], Config.pos_epsilon, None)
            return {"A": domain_A, "B": domain_B}
        else:
            domain_A = np.clip(self.data[idx, 0:5], Config.pos_epsilon, None)
            return {"A": domain_A}

# 数据预处理
def preprocess_data(train_path, test_path):

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = train_df.iloc[:, :-Config.group_sizes[-1]]
    train_labels = train_df.iloc[:, -Config.group_sizes[-1]:]#1:]#
   
    train_labels_tmp=np.array(train_df.iloc[:, -1])
    train_labels_tmp.sort()
    max_train_label_value=train_labels_tmp[-1]
    print(max_train_label_value)

    scaler.fit(pd.concat([train_features, train_labels], axis=1))

    train_scaled =scaler.transform(train_df)#scaler.transform(train_df)#train_df#s
    train_df = pd.DataFrame(train_scaled, columns=train_df.columns)

    test_scaled = scaler.transform(test_df)#test_df#
    test_df = pd.DataFrame(test_scaled, columns=test_df.columns)
    
    return train_df, test_df, scaler,max_train_label_value


def train_and_validate(trainpath,testpath,saveFolderPath):

    train_df, test_df, scaler,max_train_label_value = preprocess_data(
        trainpath,
        testpath
    )

    Config.input_dim_A = 5
    Config.input_dim_B = Config.group_sizes[-1]

    kf = KFold(n_splits=Config.n_folds, shuffle=True,random_state=0)
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n=== Fold {fold+1}/{Config.n_folds} ===")
        train_dataset = ClinicalDataset(train_df.iloc[train_idx])
        val_dataset = ClinicalDataset(train_df.iloc[val_idx])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.batch_size,
            shuffle=False
        )
        model = CycleGAN().to(Config.device)
        best_val_loss = float('inf')
        best_val_mae_rmse = float('inf')

        for epoch in range(Config.epochs):
            model.train()
            train_loss = {"G_loss": 0, "D_loss": 0, "cycle_loss": 0}
            
            for batch in train_loader:
                real_A = batch["A"].to(Config.device)
                real_B = batch["B"].to(Config.device)
                losses = model.train_step(real_A, real_B)
                train_loss["G_loss"] += losses["G_loss"]
                train_loss["D_loss"] += losses["D_loss"]
                train_loss["cycle_loss"] += losses["cycle_loss"]

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    real_A = batch["A"].to(Config.device)
                    real_B = batch["B"].to(Config.device)
                    
                    fake_B = model.G_A2B(real_A)
                    cycle_A = model.G_B2A(fake_B)
                    val_loss +=model.criterion_cycle(fake_B, real_B).item()+model.criterion_cycle(cycle_A, real_A).item()
            

            train_loss = {k: v/len(train_loader) for k, v in train_loss.items()}
            val_loss /= len(val_loader)
            print(f"\nEpoch {epoch+1}/{Config.epochs} | "
                  f"Train G: {train_loss['G_loss']:.4f} D: {train_loss['D_loss']:.4f} | "
                  f"Val Cycle: {val_loss:.4f}")

            cycle_A_mae = mean_absolute_error(real_A.cpu().numpy(), cycle_A.cpu().numpy())
            cycle_A_rmse = np.sqrt(mean_squared_error(real_A.cpu().numpy(), cycle_A.cpu().numpy()))
            fake_B_mae = mean_absolute_error(real_B.cpu().numpy(), fake_B.cpu().numpy())
            fake_B_rmse = np.sqrt(mean_squared_error(real_B.cpu().numpy(), fake_B.cpu().numpy()))
            print(f"cycle_A_ Val Results: | "
                  f"cycle_A_ MAE: {cycle_A_mae:.4f} |"
                  f"cycle_A_RMSE: {cycle_A_rmse:.4f} |"
                  f"fake_B_Val Results: | "
                  f"fake_B_ MAE: {fake_B_mae:.4f} |"
                  f"fake_B_RMSE: {fake_B_rmse:.4f}")


            total_mae_rmse=val_loss+fake_B_mae+fake_B_rmse
            if val_loss < best_val_loss:
                best_val_mae_rmse=total_mae_rmse
                best_val_loss = val_loss
                best_epoch = epoch+1
                torch.save(model.state_dict(), os.path.join(saveFolderPath,f"best_model_fold{fold}.pth"))

        print(colored(f"Best_epoch: {best_epoch}", "red"))
        best_models.append(f"best_model_fold{fold}.pth")
    
    return best_models, scaler,max_train_label_value


def evaluate_testset(testpath,model_paths, scaler,saveFolderPath,max_train_label_value):
    test_df = pd.read_excel(testpath)
    test_scaled = scaler.transform(test_df)
    test_dataset = ClinicalDataset(pd.DataFrame(test_scaled), is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    models = []
    model_paths=os.path.join(saveFolderPath,best_models[0])
    print(model_paths)
    for i in range(5):
        path=os.path.join(saveFolderPath,best_models[i])
        model = CycleGAN().to(Config.device)
        model.load_state_dict(torch.load(path), strict = True)
        model.eval()
        models.append(model)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            real_A = batch["A"].to(Config.device)
            fold_preds = []
            for model in models:
                fake_B = model.G_A2B(real_A)
                fold_preds.append(fake_B.cpu().numpy())

            avg_pred = np.mean(fold_preds, axis=0)
            all_preds.append(avg_pred)
    print(fake_B)
    pred_B = np.concatenate(all_preds, axis=0)


    dummy = np.zeros((len(test_df), 6))
    dummy[:, 0:5] = test_scaled[:, 0:5]
    dummy[:, -1:] = pred_B
    reconstructed = scaler.inverse_transform(dummy)

    preds =  reconstructed[:, -1:]
    reconstructed[:, -1:]=np.clip(preds, Config.pos_epsilon, max_train_label_value)
    result_df = pd.DataFrame(reconstructed, columns=test_df.columns[:])
    result_df.to_excel(os.path.join(saveFolderPath,"predictions.xlsx"), index=False)
import time
if __name__ == "__main__":
    Generation_task="SCC-A"#"CyFra21-1"#"NSE"#"CA72-4"#"HE4"#"CA19-9"#"CA15-3"#"CA125"#"CEA"#"AFP"#
    saveFolderPath="./save_model_generate/CycleGAN_"+Generation_task+"_transform_MinMaxScaler_"+time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())
    os.makedirs(saveFolderPath, exist_ok=True)
    trainpath="./xlsx/zhuanyi-generate-"+Generation_task+"-train.xlsx"
    testpath="./xlsx/zhuanyi-generate-"+Generation_task+"-test.xlsx"
    best_models, scaler,max_train_label_value = train_and_validate(trainpath,testpath,saveFolderPath)
    evaluate_testset(testpath,best_models, scaler,saveFolderPath,max_train_label_value)


