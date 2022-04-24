import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from read_df import df_test
from models import AE_01
from models import AE_02
from models import E_01

if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name())
    device = 'cuda'
else:
    sys.exit("I want a cup of milk tea. Thanks!")

train_dataloader = DataLoader(
        df_test(train=True),
        batch_size = 1,
        # batch_size = 8,
        shuffle = True,
        num_workers=4
        )

test_dataloader = DataLoader(
        df_test(train=False),
        batch_size = 1,
        shuffle = False,
        num_workers=4
        )

# model = AE_01()
path = '/home/roy/roy/4_AutoEncoder/pretrained/df48_E01.pt'
model = LSTM_05()
model.to(device)
model.double()
# optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
# judge = torch.nn.MSELoss()
cpt = torch.load(path)
model.load_state_dict(cpt['model_state_dict'])
optimizer.load_state_dict(cpt['optimizer_state_dict'])
model.eval()

pred_list = []
hid_list = []
for x, y in test_dataloader:
    pred, hid = model(x.cuda())
    pred_list.append(pred.detach().cpu().numpy()[0])
    hid_list.append(hid.detach().cpu().numpy()[0])

# print(np.asarray( pred_list ))
# print("pred_list: ", pred_list)
np.savetxt("1015pretrained_pred_E01.csv", np.asarray( pred_list ), delimiter=',')
np.savetxt("1015pretrained_hid_E01.csv", np.asarray( hid_list ), delimiter=',')
