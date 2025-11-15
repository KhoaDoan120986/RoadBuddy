from dataloader import VideoTextDataset
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler
from model import VideoQAModel
import torch
import torch.nn as nn
from tqdm import tqdm
import os 
import argparse
import logging
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
import gc 
import json
import datetime


global logger
args = None 
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_feature_path', type=str, required=True)
    parser.add_argument('--text_feature_path', type=str, required=True)
    parser.add_argument('--train_text_metadata_path', type=str, required=True)
    parser.add_argument('--val_text_metadata_path', type=str, required=True)
    parser.add_argument('--test_video_feature_path', type=str, required=True)
    parser.add_argument('--test_text_feature_path', type=str, required=True)
    parser.add_argument('--test_text_metadata_path', type=str, required=True)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_choices', type=int, default=5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--model_dim', type=int, default=2560)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.n_gpus = torch.distributed.get_world_size()

    return args

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def parse_batch(batch, device):
    video_ids, video_feats, video_masks, text_feats, text_attention_masks, answer_vecs = batch

    video_feats = video_feats.to(device, non_blocking=True)  # (B, S, D)
    video_masks = video_masks.to(device, non_blocking=True)
    text_feats = text_feats.to(device, non_blocking=True)  # (B, N, D)
    text_attention_masks = text_attention_masks.to(device, non_blocking=True)
    answer_vecs = answer_vecs.to(device, non_blocking=True)

    return video_ids, video_feats, video_masks, text_feats, text_attention_masks, answer_vecs

def train(model, dataloader, optimizer, args, device, scheduler=None):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    if args.local_rank == 0:
        pbar = tqdm(dataloader, total=len(dataloader), desc="Training", leave=False)
    else:
        pbar = dataloader

    for step, batch in enumerate(pbar):
        _, video_feats, video_masks, text_feats, text_attention_masks, answer_vecs = parse_batch(batch, device)
        targets = answer_vecs.argmax(dim=1)

        logits = model(video_feats, text_feats, video_masks, text_attention_masks)
        loss = criterion(logits, targets)
        loss = loss / args.gradient_accumulation_steps

        loss.backward()
        total_loss += loss.detach().item() * args.gradient_accumulation_steps

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        avg_loss_so_far = total_loss / (step + 1)
        if args.local_rank == 0:
            pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})

        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        video_ids, video_feats, video_masks, text_feats, text_attention_masks, answer_vecs = parse_batch(batch, device)
        targets = answer_vecs.argmax(dim=1)

        logits = model(video_feats, text_feats, video_masks, text_attention_masks)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def prep_optimizer(model, len_train_iter, args, device):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'norm.weight', 'norm.bias']

    decay_params = [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_training_steps = int(len_train_iter / args.gradient_accumulation_steps) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=False)

    return optimizer, lr_scheduler, model


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, best=False, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"best_checkpoint.pt" if best else f"epoch_{epoch+1}.pt"
    save_path = os.path.join(save_dir, filename)

    # If using DataParallel or DDP, save the underlying model
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path="checkpoints/best_checkpoint.pt", device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.4f})")
    return checkpoint

def inference(ckpt_path, dataloader, args, device):
    model = VideoQAModel(
        embed_dim=args.model_dim, 
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    ).to(device)

    load_checkpoint(model, checkpoint_path=ckpt_path, device=device)
    model.eval()

    results = {}
    for batch in tqdm(dataloader, desc="Inference", leave=False):
        video_ids, video_feats, video_masks, text_feats, text_attention_masks, _ = parse_batch(batch, device)

        with torch.no_grad():
            logits = model(video_feats, text_feats, video_masks, text_attention_masks)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        for vid, pred in zip(video_ids, preds):
            results[vid] = pred

    return results

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main():
    global args, logger
    args = get_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    logger = get_logger(filename="log.txt")

    if args.do_train:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if args.local_rank == 0:
            logging.info("Starting training with the following hyperparameters:")
            for arg in vars(args):
                logging.info(f"{arg}: {getattr(args, arg)}")


        train_dataset = VideoTextDataset(
            video_feature_path=args.video_feature_path, 
            text_feature_path=args.text_feature_path, 
            text_metadata_path=args.train_text_metadata_path, 
            max_choices=args.max_choices
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            drop_last=True
        )

        if args.local_rank == 0:
            logging.info(f"Max video tokens: {train_dataset.max_video_tokens}")
            logging.info(f"Max text tokens: {train_dataset.max_video_tokens}")
            logging.info(f"Number of train samples: {len(train_dataset)}")
            
            val_dataset = VideoTextDataset(
                video_feature_path=args.video_feature_path, 
                text_feature_path=args.text_feature_path, 
                text_metadata_path=args.val_text_metadata_path, 
                max_choices=args.max_choices
            )
            val_sampler = SequentialSampler(val_dataset)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.test_batch_size,
                sampler=val_sampler
            )
            logging.info(f"Number of validation samples: {len(val_dataset)}")

        model = VideoQAModel(
            embed_dim=args.model_dim, 
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads, 
            num_layers=args.num_layers, 
            dropout=args.dropout
        ).to(device)

        optimizer, lr_scheduler, model = prep_optimizer(model, len(train_loader), args, device)

        if args.local_rank == 0:
            parameter_number = get_parameter_number(model)
            logger.info("The model's parameter number: {}".format(parameter_number))
            logger.info("Starting training...")

        best_val_acc = 0.0
        for epoch in range(args.num_epochs):
            train_loader.sampler.set_epoch(epoch)
            train_loss = train(model, train_loader, optimizer, args, device, scheduler=lr_scheduler)
            if args.local_rank == 0:    
                val_loss, val_acc = evaluate(model, val_loader, device)
                current_lr = lr_scheduler.get_last_lr()[0]
                
                logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] | LR={current_lr:.6f} | "
                            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

                save_checkpoint(model, optimizer, lr_scheduler, epoch, val_acc, best=False)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, val_acc, best=True)
                    logger.info(f"New best model saved with Val Acc = {val_acc:.4f}")


        del model, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    if args.local_rank == 0:
        test_dataset = VideoTextDataset(
            video_feature_path=args.test_video_feature_path,
            text_feature_path=args.test_text_feature_path,
            text_metadata_path=args.test_text_metadata_path,
            max_choices=args.max_choices,
        )
        test_sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size,
            sampler=test_sampler
        )
        ckpt_path = "checkpoints/best_checkpoint.pt"
        results = inference(ckpt_path, test_loader, args, device)
        logger.info(f"Inference complete on test set. Total samples: {len(results)}")
        with open("test_results.json", "w", encoding="utf-8") as f_out:
            json.dump(results, f_out)

if __name__ == "__main__":
    main()