import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse

from model import StudentModel
from dataset import GeometryStudentDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/geometry_labels.jsonl")
    parser.add_argument("--esm_path", type=str, default="pretrained/esm2_t12_35M_UR50D")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Dataset & Dataloader
    print("Initializing Dataset...")
    dataset = GeometryStudentDataset(
        jsonl_path=args.data_path,
        esm_model_path=args.esm_path
    )
    
    # Split Train/Val (Simple 90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Model
    print("Initializing Model...")
    model = StudentModel(esm_model_path=args.esm_path)
    model.to(args.device)
    
    # 3. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore non-pocket residues
    
    # 4. Training Loop
    print("Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            ligand_feat = batch['ligand_feat'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask, ligand_feat)
            
            # Flatten for Loss: [Batch*Seq, Num_Classes] vs [Batch*Seq]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                ligand_feat = batch['ligand_feat'].to(args.device)
                labels = batch['labels'].to(args.device)
                
                logits = model(input_ids, attention_mask, ligand_feat)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_student.pth"))
            print("Saved Best Model!")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
