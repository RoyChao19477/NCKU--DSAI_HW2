import sys
from tabnanny import verbose
from tkinter import E
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from read_df import df
from models import CGAN_D, CGAN_G

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

lr_G = 0.00001
lr_D = 0.00001
b1 = 0.9
b2 = 0.999
n_epoch = 600

G = CGAN_G(4)
D = CGAN_D(1)
G.to(device)
D.to(device)
G.double()
D.double()

optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(b1, b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_G, betas=(b1, b2))

adversarial_loss = torch.nn.BCELoss()
L1 = torch.nn.L1Loss()

t_best = 100000
try:
    for i in range(n_epoch):
        total_loss = 0.0

        for x, y in train_dataloader:
            #y = y.double()

            """
            if device=='cpu':
                pred = model(x) * x[-1][0]
                loss = judge(pred, y)
            else:
                pred = model(x.cuda() * x[-1][0])
                loss = judge(pred, y.cuda())
            """

            G.eval()
            D.train()
            optimizer_D.zero_grad()
            #real = D(x.cuda().float())
            #d_real_loss = (adversarial_loss(real, torch.tensor([[1.0]]).cuda()) ) # + auxiliary_loss(real, torch.tensor([[1.0]]).cuda())) / 2
            real = D(y.double())
            d_real_loss = (adversarial_loss(real, torch.tensor(1.0).double()) ) # + auxiliary_loss(real, torch.tensor([[1.0]]).cuda())) / 2

            #fake_G = G(torch.tensor( np.random.normal(0, 10, (4, 1))).float().cuda()) * x[-1][0]
            #fake_G = G( (1 + 0.1 * torch.tensor( np.random.normal(0, 1, (4))).double()) ) * x[-1][0]
            fake_G = G(x.squeeze(0).double()) * x[-1][0].double()
            fake = D( fake_G )
            #d_fake_loss = (adversarial_loss(fake, torch.tensor([[0.0]]).cuda()) ) # + auxiliary_loss(fake,torch.tensor([[0.0]]).cuda())) / 2
            d_fake_loss = (adversarial_loss(fake, torch.tensor(0.0).double()) ) # + auxiliary_loss(fake,torch.tensor([[0.0]]).cuda())) / 2

            D_loss = (d_real_loss + d_fake_loss) / 2

            D_loss.backward()
            optimizer_D.step()
            
            G.train()
            D.eval()            
            optimizer_G.zero_grad()
            #z = torch.tensor( np.random.normal(0, 10, (4))).float()
            #g = G(z.cuda()) * x[-1][0]
            g = G(x.squeeze(0).double()) * x[-1][0].double()
            val = D(g)
            #G_loss = adversarial_loss(val, torch.tensor([[1.0]]).cuda())
            G_loss = adversarial_loss(val, torch.tensor(1.0).double())
            G_loss.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (i, n_epoch, D_loss.item(), G_loss.item())
        )

        G.eval()
        pred_list = []
        test_loss = 0
        for x, y in test_dataloader:
            if device=='cpu':
                pred = G(x.squeeze(0).double()) * x[-1][0].double()
            else:
                pred = G(x.squeeze(0).double().cuda() * x[-1][0])
            pred_list.append(pred.detach().cpu().numpy()[0])

            if device=='cpu':
                loss = L1(pred, y)
            else:
                loss = L1(pred, y.cuda())
            
            test_loss += loss.item()
    
        print("Predict loss: ", test_loss)
        print("     pred_list: ", pred_list)

        if test_loss < t_best:
            t_best = test_loss
            torch.save({ 
                    'model_state_dict': G.state_dict(), 
                    'optimizer_state_dict': optimizer_G.state_dict()}, 'results/5_GAN_G.pt')
            torch.save({ 
                'model_state_dict': D.state_dict(), 
                'optimizer_state_dict': optimizer_D.state_dict()}, 'results/5_GAN_D.pt')
        
            np.savetxt("results/5_GAN.csv", np.asarray( pred_list ), delimiter=',')
                        
except Exception as e:
    exception = e
    raise
