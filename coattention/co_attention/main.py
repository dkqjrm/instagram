import pickle
import numpy as np
import torch
import argparse
import os
import time
import random
import torch.nn as nn
import shutil
import io
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import create_raw_data, HashtagDataset, hashcollator
from config import Configuration
from model import Co_Attention
from torch.optim import Adam

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def scoring(pred, target, topk):
    pred = torch.argsort(pred, dim=1, descending=True)
    pred = pred.cpu().detach().numpy()  # [batch_size, hashtag_vocab_size]
    target = target.numpy()  # [batch_size, hashtag_vocab_size]
    tag_label = []
    for this_data in target:
        tag_label.append([])
        for idx, each_tag in enumerate(this_data):
            if each_tag != 0:
                tag_label[-1].append(idx)

    precision = []
    recall = []
    f1 = []
    for i in range(len(pred)):
        this_precision = 0
        this_recall = 0
        this_f1 = 0
        for j in range(topk):
            if pred[i][j] in target[i]:
                this_precision += 1
        for j in range(len(target[i])):
            if target[i][j] in pred[i][:topk]:
                this_recall += 1
        this_precision /= topk
        this_recall /= len(target[i])
        if this_precision != 0 and this_recall != 0:
            this_f1 = 2 * (this_precision * this_recall) / (this_precision + this_recall)
        precision.append(this_precision)
        recall.append(this_recall)
        f1.append(this_f1)
    return precision, recall, f1

def save_checkpoint(state, is_best, model_save_path, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_path, 'model_best.pth.tar'))

if __name__ == "__main__":
    np.random.seed(42)
    torch.random.manual_seed(42)
    parser = argparse.ArgumentParser(description='Co-attention')
    parser.add_argument('--data_path', type=str, default='./data/data/')
    parser.add_argument('--emoji', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--maxlen', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./saved_model/')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--e_dim', type=int, default=300)
    parser.add_argument('--grid', type=int, default=49)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--stopping_threshold', type=int, default=10)
    args = parser.parse_args()
    save_path = os.path.join(args.save_path,
                             time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    os.makedirs(save_path)
    cfg = Configuration(vars(args), save_path)
    writer = SummaryWriter(log_dir=save_path)

    # data 처음 만들 시 실행
    # all_data = create_raw_data(cfg.data_path, cfg.emoji)
    with open('raw_data.pkl', 'rb') as f:
        raw_img, raw_text, raw_hashtags, hashtag_vocab, word_vocab, hash_onehot = CPU_Unpickler(f).load()

    # with open('raw_data.pkl', 'rb') as f:
    #     raw_img, raw_text, raw_hashtags, hashtag_vocab, word_vocab, hash_onehot = pickle.load(f)

    shuffled_list = [q for q in range(len(raw_img))]
    random.shuffle(shuffled_list)
    train_img = []
    train_txt = []
    train_tags = []
    train_tags_onehot = []
    val_img = []
    val_txt = []
    val_tags = []
    val_tags_onehot = []

    for q in shuffled_list[:5000]:
        train_img.append(raw_img[q])
        train_txt.append(raw_text[q])
        train_tags.append(raw_hashtags[q])
        train_tags_onehot.append(hash_onehot[q])
    for q in shuffled_list[5000:]:
        val_img.append(raw_img[q])
        val_txt.append(raw_text[q])
        val_tags.append(raw_hashtags[q])
        val_tags_onehot.append(hash_onehot[q])

    train_dataset = HashtagDataset(train_img, train_txt, train_tags, train_tags_onehot)
    val_dataset = HashtagDataset(val_img, val_txt, val_tags, val_tags_onehot)

    trn_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=hashcollator, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, collate_fn=hashcollator)

    # for q in range(len(train_dataset)):
    #     print(train_dataset.img[q])
    #     print(train_dataset.text[q])
    #     print(train_dataset.hashtags[q])

    model = Co_Attention(cfg.e_dim, cfg.grid, len(word_vocab), len(hashtag_vocab), cfg.k).to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    global_steps = 0
    epoch = 0
    max_f1 = 0
    stop_cnt = 0
    while True:
        epoch += 1
        model.train()
        precision = []
        recall = []
        f1 = []
        train_loss = 0
        cnt = 0
        for batch in tqdm(trn_dataloader, total=len(trn_dataloader)):
            img = batch[0].to(cfg.device)
            text = batch[1].to(cfg.device)
            hashtags = batch[2].to(cfg.device)
            hashtags_onehot = batch[3]
            logits = model(img, text)  # [batch_size, hashtag_vocab_size]
            loss = criterion(logits, hashtags)
            train_loss += loss.item()
            cnt += len(img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_p, batch_r, batch_f1 = scoring(logits, hashtags_onehot, cfg.topk)
            precision.extend(batch_p)
            recall.extend(batch_r)
            f1.extend(batch_f1)

            writer.add_scalar(tag='batch_precision',
                              scalar_value=sum(batch_p)/len(batch_p),
                              global_step=global_steps)
            writer.add_scalar(tag='batch_recall',
                              scalar_value=sum(batch_r) / len(batch_r),
                              global_step=global_steps)
            writer.add_scalar(tag='batch_f1',
                              scalar_value=sum(batch_f1) / len(batch_f1),
                              global_step=global_steps)
            writer.add_scalar(tag='batch_loss',
                              scalar_value=loss.item(),
                              global_step=global_steps)
            global_steps += 1

        writer.add_scalar(tag='train_precision',
                          scalar_value=sum(precision) / len(precision),
                          global_step=epoch)
        writer.add_scalar(tag='train_recall',
                          scalar_value=sum(recall) / len(recall),
                          global_step=epoch)
        writer.add_scalar(tag='train_f1',
                          scalar_value=sum(f1) / len(f1),
                          global_step=epoch)
        writer.add_scalar(tag='train_loss',
                          scalar_value=train_loss / cnt,
                          global_step=epoch)

        model.eval()
        precision = []
        recall = []
        f1 = []
        val_loss = 0
        cnt = 0
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            img = batch[0].to(cfg.device)
            text = batch[1].to(cfg.device)
            hashtags = batch[2].to(cfg.device)
            hashtags_onehot = batch[3]
            with torch.no_grad():
                logits = model(img, text)  # [batch_size, hashtag_vocab_size]
            loss = criterion(logits, hashtags)
            val_loss += loss.item()
            cnt += len(img)
            batch_p, batch_r, batch_f1 = scoring(logits, hashtags_onehot, cfg.topk)
            precision.extend(batch_p)
            recall.extend(batch_r)
            f1.extend(batch_f1)

        val_p = sum(precision)/len(precision)
        val_r = sum(recall) / len(recall)
        val_f1 = sum(f1) / len(f1)
        writer.add_scalar(tag='val_precision',
                          scalar_value=val_p,
                          global_step=epoch)
        writer.add_scalar(tag='val_recall',
                          scalar_value=val_r,
                          global_step=epoch)
        writer.add_scalar(tag='val_f1',
                          scalar_value=val_f1,
                          global_step=epoch)
        writer.add_scalar(tag='val_loss',
                          scalar_value=val_loss / cnt,
                          global_step=epoch)

        if val_f1 > max_f1:
            max_f1 = val_f1
            stop_cnt = 0
            is_best = True
        else:
            stop_cnt += 1
            is_best = False

        save_checkpoint({
            'epoch': epoch,
            'model': model,
            'state_dict': model.state_dict(),
            'precision': val_p,
            'recall': val_r,
            'f1-score': val_f1,
            'optimizer': optimizer.state_dict()
        }, is_best, save_path, os.path.join(save_path, 'epoch' + str(epoch) + '.pth.tar'))

        if stop_cnt > cfg.threshold:
            print("Training finished.")
            break
