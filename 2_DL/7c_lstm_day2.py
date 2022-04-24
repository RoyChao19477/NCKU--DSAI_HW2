import sys
from tabnanny import verbose
from tkinter import E
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, recall_score

from read_df import df_day2
from models import L_05_day2_C

if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name())
    device = 'cuda'
else:
    device = 'cpu'
    #sys.exit("I want a cup of milk tea. Thanks!")

train_dataloader = DataLoader(
        df_day2(train=True),
        batch_size = 1,
        # batch_size = 8,
        shuffle = False,
        num_workers=0
        )

test_dataloader = DataLoader(
        df_day2(train=False),
        batch_size = 1,
        shuffle = False,
        num_workers=0
        )

model = L_05_day2_C(4)
model.to(device)
model.double()

optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0)
judge = torch.nn.L1Loss()
L2 = torch.nn.MSELoss()
CEL = torch.nn.CrossEntropyLoss()
adversarial_loss = torch.nn.BCELoss()

t_best = 100000
try:
    for i in range(1000):
        model.train()
        total_loss = 0.0

        for x, y in train_dataloader:
            y = y.double()

            if device=='cpu':
                pred = model(x) * x[-1][0]
                diff_a = (pred[:,1] - pred[:,0])
                diff_b = y[:,1] - y[:,0]
                loss = L2(pred, y) + L2(pred[0], y[0])
            else:
                pred = model(x.cuda() * x[-1][0])
                y = y.cuda()
                diff_a = (pred[:,1] - pred[:,0])
                diff_b = y[:,1] - y[:,0]
                loss = judge(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print("Training: ", i, " traing loss: ", total_loss)

        model.eval()
        y_list = []
        pred_list = []
        test_loss = 0
        for x, y in test_dataloader:
            if device=='cpu':
                pred = model(x) * x[-1][0]
            else:
                pred = model(x.cuda() * x[-1][0])
            
            y_list.append(y[0].detach().cpu().numpy())
            pred_list.append(pred[0].detach().cpu().numpy())

            if device=='cpu':
                diff_a = (pred[:,1] - pred[:,0])
                diff_b = y[:,1] - y[:,0]
                loss = judge(pred, y)
            else:
                y = y.cuda()
                diff_a = (pred[:,1] - pred[:,0])
                diff_b = y[:,1] - y[:,0]
                loss = judge(pred, y)
            
            test_loss += loss.item()
    
        print("Predict loss: ", test_loss)
        print("     pred_list: ", pred_list)
        
        y_choose = []
        pred_choose = []
        for i in range(len(y_list)):
            if y_list[i][1] - y_list[i][0] > 0:
                y_choose.append(1)
            else:
                y_choose.append(0)
                
            if pred_list[i][1] - pred_list[i][0] > 0:
                pred_choose.append(1)
            else:
                pred_choose.append(0)

        test_loss += 1 / accuracy_score(y_choose, pred_choose)

        if test_loss < t_best:
            t_best = test_loss
            torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()}, 'results/7c_lstm_day2.pt')
            np.savetxt("results/7c_lstm_day2.csv", np.asarray( pred_list ), delimiter=',')
                        
except Exception as e:
    exception = e
    raise
