import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

from model import StudentModel
from dataset import GeometryStudentDataset

def calculate_accuracy(logits, labels):
    # logits: [N, Num_Classes]
    # labels: [N]
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/geometry_labels.jsonl")
    parser.add_argument("--esm_path", type=str, default="pretrained/esm2_t12_35M_UR50D")
    parser.add_argument("--output_dir", type=str, default="checkpoints_improved")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of ESM layers to unfreeze")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Setup Logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Starting training with args: {args}")
    
    # 1. Dataset & Dataloader
    logging.info("Initializing Dataset...")
    dataset = GeometryStudentDataset(
        jsonl_path=args.data_path,
        esm_model_path=args.esm_path
    )
    
    # Split Train/Val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Model
    logging.info("Initializing Model...")
    model = StudentModel(esm_model_path=args.esm_path)
    model.to(args.device)
    
    # Unfreeze last N layers of ESM
    if args.unfreeze_layers > 0:
        logging.info(f"Unfreezing last {args.unfreeze_layers} layers of ESM-2...")
        for param in model.esm.encoder.layer[-args.unfreeze_layers:].parameters():
            param.requires_grad = True
        # Also unfreeze the final layer norm and pooler if they exist
        for param in model.esm.encoder.emb_layer_norm_after.parameters():
            param.requires_grad = True
            
    # 3. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            ligand_feat = batch['ligand_feat'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask, ligand_feat)
            
            # Flatten for Loss
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            
            loss = criterion(flat_logits, flat_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            c, t = calculate_accuracy(flat_logits, flat_labels)
            train_correct += c
            train_total += t
            
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                ligand_feat = batch['ligand_feat'].to(args.device)
                labels = batch['labels'].to(args.device)
                
                logits = model(input_ids, attention_mask, ligand_feat)
                
                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels.view(-1)
                
                loss = criterion(flat_logits, flat_labels)
                val_loss += loss.item()
                
                c, t = calculate_accuracy(flat_logits, flat_labels)
                val_correct += c
                val_total += t
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "best_student.pt")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved best model to {save_path}")
            
        # Save Last Model
        torch.save(model.state_dict(), os.path.join(args.output_dir, "last_student.pt"))

if __name__ == "__main__":
    main()
