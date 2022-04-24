import sys
from tabnanny import verbose
from tkinter import E
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from read_df import df
from models import L_05

if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name())
    device = 'cuda'
else:
    device = 'cpu'
    #sys.exit("I want a cup of milk tea. Thanks!")

train_dataloader = DataLoader(
        df(train=True),
        batch_size = 1,
        # batch_size = 8,
        shuffle = False,
        num_workers=0
        )

test_dataloader = DataLoader(
        df(train=False),
        batch_size = 1,
        shuffle = False,
        num_workers=0
        )

model = L_05(4)
model.to(device)
model.double()

optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0)
judge = torch.nn.L1Loss()

t_best = 100000
try:
    for i in range(1000):
        model.train()
        total_loss = 0.0

        for x, y in train_dataloader:
            y = y.double()

            if device=='cpu':
                pred = model(x) * x[-1][0]
                loss = judge(pred, y)
            else:
                pred = model(x.cuda() * x[-1][0])
                loss = judge(pred, y.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print("Training: ", i, " traing loss: ", total_loss)

        model.eval()
        pred_list = []
        test_loss = 0
        for x, y in test_dataloader:
            if device=='cpu':
                pred = model(x) * x[-1][0]
            else:
                pred = model(x.cuda() * x[-1][0])
            pred_list.append(pred.detach().cpu().numpy()[0])

            if device=='cpu':
                loss = judge(pred, y)
            else:
                loss = judge(pred, y.cuda())
            
            test_loss += loss.item()
    
        print("Predict loss: ", test_loss)
        print("     pred_list: ", pred_list)

        if test_loss < t_best:
            t_best = test_loss
            torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}, 'results/1_lstm.pt')
            np.savetxt("results/1_lstm.csv", np.asarray( pred_list ), delimiter=',')
                        
except Exception as e:
    exception = e
    raise
