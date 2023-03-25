from model.VQ_VAE import VQ_VAE
from data.data import ImageSet
import random
import numpy as np
import torch
import os, shutil
from copy import deepcopy
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn

def set_seed(seed = 1008):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def build_dir(log_path, save_path):
    try:
        os.mkdir(log_path)
    except Exception as e:
        pass

    try:
        os.mkdir(save_path)
    except Exception as e:
        pass

def train_model(model_path, dataset, batch_size, num_epochs, 
                        lr, device, log_file):
    
    model = torch.load(model_path).to(device)
    
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    best_train = 1e10
    
    for epoch in trange(num_epochs):
        log_file.write(str(epoch)+':\n')

        total_train_loss = 0

        dataloader = DataLoader(dataset, batch_size = batch_size, 
                                shuffle = True, drop_last = True)
        for input_img in dataloader:
            input_img = input_img.to(device)
            output = model(input_img)
            
            batch_loss = criterion(output, input_img)
            total_train_loss += batch_loss.item()

            #batch_loss = batch_loss.requires_grad_()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        if total_train_loss < best_train:
            best_train = total_train_loss
            torch.save(model, model_path)
        log_file.write('loss = ' + str(total_train_loss) + '\n')
        #print(total_train_loss)
    

def evaluate(model_path, input, device, tgt_tokenizer):
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        prompt = [0 for _ in range(200)]
        prompt[0] = tgt_tokenizer.word2id('<SOS>')
        input = input.to(device).unsqueeze(0)
        for i in range(1, 200):
            tgt = torch.tensor(prompt).unsqueeze(0).to(device)
            output = model(input, tgt, device).squeeze(0)[-1]
            pred = output.argmax(dim = -1).item()
            if(pred != 0):
                prompt[i]=pred
            else:
                break
        print(tgt_tokenizer.de_tokenize(prompt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train')
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--seed', default = 1008, type = int)
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--lr', default = 0.0001, type = float)
    parser.add_argument('--dropout', default = 0.5, type = float)
    parser.add_argument('--token_dim', default = 1280, type = int)
    parser.add_argument('--token_siz', default = 8192, type = int)
    parser.add_argument('--num_heads', default = 4, type = int)
    parser.add_argument('--medium_dim', default = 128, type = int)
    parser.add_argument('--device', default = 'cuda', type = str)
    parser.add_argument('--log_path', default = 'train_logs/', type = str)
    parser.add_argument('--save_path', default = 'model_file/', type = str)
    parser.add_argument('--resume', action = 'store_true')
    parser.add_argument('--type', default = 'train', type = str)
    parser.add_argument('--test_path', default = 'test.txt', type = str)
    args = parser.parse_args()
    
    set_seed(args.seed)
    build_dir(args.log_path, args.save_path)

    device = args.device
    
    model_path = args.save_path + 'transformer.pkl'
    
    if args.type == "test":
        
        pass
       #generate_image(model_path, args.input_dim, device, args.num_images)
    else:
        dataset = ImageSet('../data/imgs/')
        if not args.resume:
            model = VQ_VAE(args.token_siz, args.token_dim, args.medium_dim, args.dropout, args.num_heads)
            torch.save(model, model_path)
            
        log = open(args.log_path + 'train_logs. txt', 'w')

        

        train_model(model_path, dataset, batch_size = args.batch_size, 
                    num_epochs = args.epochs, lr = args.lr, device = device, log_file = log)