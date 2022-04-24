from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

from t2v import SineActivation, CosineActivation

class CNN_01(nn.Module):
    def __init__(self, input_size):
        super(CNN_01, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 48, 3)
        self.conv2 = nn.Conv1d(48, 24, 3)
        self.conv3 = nn.Conv1d(24, 12, 3)
        #self.conv4 = nn.Conv1d(12, 1, 3, padding=1)
        self.l1 = nn.Linear(12, 96)
        self.l2 = nn.Linear(96, 48)
        self.l3 = nn.Linear(48, 24)
        self.l4 = nn.Linear(24, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [N, Cin, L]
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.l1(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        # print(x.shape)
        x = x.squeeze(0)

        return x

class DNN_01(nn.Module):
    def __init__(self, input_size):
        super(DNN_01, self).__init__()

        self.l1 = nn.Linear(input_size, 96)
        self.l2 = nn.Linear(96, 48)
        self.l3 = nn.Linear(48, 24)
        self.l4 = nn.Linear(24, 1)

    def forward(self, x):
        x = self.l1(x[:,-1,:])
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)

        return x

class RNN_01(nn.Module):
    def __init__(self, input_size):
        super(RNN_01, self).__init__()
        self.input_size = input_size
        self.days = 7
        self.hid_size = 24

        self.rnn = nn.RNN(
                input_size = self.input_size,
                hidden_size = self.hid_size,
                batch_first = False,
                num_layers = 2,
                dropout = 0.2,
                bidirectional = False,
                )

        self.fc1 = nn.Linear(self.hid_size, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = x.permute(1,0,2)    # [L, N, Hin]
        h0 = torch.randn(2, 1, self.hid_size).double()
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x, _ = self.rnn(x, (h0))
        x = F.relu(x)
        x = self.fc1(x[-1,:,:])
        x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, 0.2)
        #x = torch.squeeze(x, 0)
    
        return x

class FCN_01(nn.Module):
    def __init__(self, input_size):
        super(FCN_01, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 48, 3)
        self.conv2 = nn.Conv1d(48, 24, 3)
        self.conv3 = nn.Conv1d(24, 1, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [N, Cin, L]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = x.squeeze(0)
        # print(x.shape)
        # efsfs

        return x



class LSTM_05(nn.Module):
    def __init__(self, input_size):
        super(LSTM_05, self).__init__()
        self.input_size = input_size
        self.days = 1
        self.hid_size = 100

        self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hid_size,
                batch_first = False,
                num_layers = 2,
                dropout = 0.2,
                bidirectional = False,
                )
        
        self.fc1 = nn.Linear(self.hid_size, 12)
        self.fc2 = nn.Linear(12, 1)

        self.last = nn.Sequential(
            nn.Linear(self.hid_size, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(50, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        self.conv = nn.Conv1d(7, 1, 1)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1,0,2)    # [L, N, Hin]
        x, _ = self.lstm(x)
        x = F.leaky_relu(x, 0.2)
        x = self.last(x)
        x = x.squeeze(0).squeeze(0)
        return x

class BLSTM_05(nn.Module):
    def __init__(self, input_size):
        super(BLSTM_05, self).__init__()
        self.input_size = input_size
        self.days = 7
        self.hid_size = 24

        self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hid_size,
                batch_first = False,
                num_layers = 4,
                dropout = 0.2,
                bidirectional = True,
                )
        
        self.fc1 = nn.Linear(self.hid_size, 12)
        self.fc2 = nn.Linear(12, 1)

        self.last = nn.Sequential(
            nn.Linear(96, 48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        self.conv = nn.Conv1d(7, 1, 1)

    def forward(self, x):
        x = x.permute(1,0,2)    # [L, N, Hin]
        x, (h_n, h_c) = self.lstm(x)
        x = F.leaky_relu(h_n.view(1, -1), 0.2)
        x = self.last(x)

        return x

class LLSTM_05(nn.Module):
    def __init__(self, input_size):
        super(LLSTM_05, self).__init__()
        self.input_size = input_size
        self.days = 7
        self.hid_size = 24

        self.first = nn.Sequential(
            nn.Conv1d(input_size, self.hid_size, 3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(self.hid_size, self.hid_size, 3, dilation=2),
            nn.ReLU(),
            #nn.Conv1d(self.hid_size, self.hid_size, 3, dilation=4),
            #nn.ReLU(),
        )
        self.lstm = nn.LSTM(
                input_size = self.hid_size,
                hidden_size = self.hid_size,
                batch_first = False,
                num_layers = 2,
                dropout = 0.2,
                bidirectional = False,
                )
        
        

        self.fc1 = nn.Linear(self.hid_size, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = x.permute(1,0,2)    # [L, N, Hin]
        x = x.permute(1,2,0)    # [N, Hin, L]
        x = self.first(x)
        
        x = x.permute(2,0,1)    # [L, N, Hin]
        h0 = torch.randn(2, 1, self.hid_size).double()
        c0 = torch.randn(2, 1, self.hid_size).double()
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x, (h_n, h_c) = self.lstm(x, (h0, c0))
        x = F.relu(x)
        x = self.fc1(x[-1,:,:])
        x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, 0.2)
        #x = torch.squeeze(x, 0)
        #print(x.shape)
        #esfs
        return x

class LLLSTM_05(nn.Module):
    def __init__(self, input_size):
        super(LLLSTM_05, self).__init__()
        self.input_size = input_size
        self.days = 7
        self.hid_size = 24

        self.first = nn.Sequential(
            nn.Conv1d(input_size, self.hid_size*2, 3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(self.hid_size*2, self.hid_size*2, 3, dilation=2),
            nn.ReLU(),
            #nn.Conv1d(self.hid_size, self.hid_size, 3, dilation=4),
            #nn.ReLU(),
        )
        self.lstm = nn.LSTM(
                input_size = self.hid_size*2,
                hidden_size = self.hid_size,
                batch_first = False,
                num_layers = 2,
                dropout = 0.2,
                bidirectional = False,
                )
        
        

        self.fc1 = nn.Linear(self.hid_size, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = x.permute(1,0,2)    # [L, N, Hin]
        x = x.permute(1,2,0)    # [N, Hin, L]
        x = self.first(x)
        
        x = x.permute(2,0,1)    # [L, N, Hin]
        h0 = torch.randn(2, 1, self.hid_size).double()
        c0 = torch.randn(2, 1, self.hid_size).double()
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x, (h_n, h_c) = self.lstm(x, (h0, c0))
        x = F.relu(x)
        x = self.fc1(x[-1,:,:])
        x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, 0.2)
        #x = torch.squeeze(x, 0)
        #print(x.shape)
        #esfs
        return x

class LLLLSTM_05(nn.Module):
    def __init__(self, input_size):
        super(LLLLSTM_05, self).__init__()
        self.input_size = input_size
        self.days = 7
        self.hid_size = 24

        self.first = nn.Sequential(
            nn.Conv1d(input_size, self.hid_size*2, 3, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.hid_size*2, self.hid_size*2, 3, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            #nn.Conv1d(self.hid_size, self.hid_size, 3, dilation=4),
            #nn.ReLU(),
        )
        self.lstm = nn.LSTM(
                input_size = self.hid_size*2,
                hidden_size = self.hid_size,
                batch_first = False,
                num_layers = 2,
                dropout = 0.2,
                bidirectional = False,
                )
        
        

        self.fc1 = nn.Linear(self.hid_size, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = x.permute(1,0,2)    # [L, N, Hin]
        x = x.permute(1,2,0)    # [N, Hin, L]
        x = self.first(x)
        
        x = x.permute(2,0,1)    # [L, N, Hin]
        h0 = torch.zeros(2, 1, self.hid_size).double()
        c0 = torch.zeros(2, 1, self.hid_size).double()
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x, (h_n, h_c) = self.lstm(x, (h0, c0))
        x = F.relu(x)
        x = self.fc1(x[-1,:,:])

        #x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.fc2(x)
        x = F.dropout(x, 0.2)
        #x = torch.squeeze(x, 0)
        #print(x.shape)
        #esfs
        return x

class Transformer_01(nn.Module):
    def __init__(self, input_size):
        super(Transformer_01, self).__init__()
        self.input_size = input_size
        self.days = 7
        self.hid_size = 24
        self.feature_size = 24

        self.l1 = SineActivation(1, self.feature_size)
        self.f1 = nn.Linear(self.feature_size, 2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=6, nhead=3)
        self.T = nn.TransformerEncoder(encoder_layer, num_layers=6)


        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        #print(x.shape)
        src = self.l1(x.permute(1,0,2)[:,:,-1])
        #print(src.shape)
        src = self.f1(src)
        src = src.unsqueeze(0)
        #print(src.shape)
        x = torch.cat([x, src], 2)
        #print(x.shape)

        x = x.permute(1,0,2)    # [L, N, Hin]
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        #print(x.shape)
        x = self.T(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.fc1(x[-1,:,:])
        #x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.dropout(x, 0.2)
        #x = torch.squeeze(x, 0)
    
        return x

class Transformer_02(nn.Module):
    def __init__(self, input_size):
        super(Transformer_02, self).__init__()
        self.input_size = input_size
        self.days = 7
        self.hid_size = 24
        self.feature_size = 24

        self.l1 = SineActivation(1, self.feature_size)
        self.f1 = nn.Linear(self.feature_size, 2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=3)
        self.T = nn.TransformerEncoder(encoder_layer, num_layers=6)


        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = x.permute(1,0,2)    # [L, N, Hin]
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        #print(x.shape)
        x = self.T(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.fc1(x[-1,:,:])
        x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, 0.2)
        #x = torch.squeeze(x, 0)
    
        return x

class Transformer_03(nn.Module):
    def __init__(self, input_size=28, hidden_layer_size=100, output_size=1):
        super(Transformer_03, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size, requires_grad=True).double(),
                            torch.zeros(1,1,self.hidden_layer_size, requires_grad=True).double())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]



class Transformer_00(nn.Module):
    def __init__(self, input_size):
        super(Transformer_00, self).__init__()
        self.input_size = input_size
        self.feature_size = 1
        self.days = 7
        self.hid_size = 24

        self.l1 = SineActivation(1, self.feature_size)
        self.f1 = nn.Linear(self.feature_size, 2)
        #self.l1 = CosineActivation(1, self.feature_size)

        self.trans = nn.Transformer(
                d_model = self.hid_size,
                nhead = 8,
                num_encoder_layers = 6,
                num_decoder_layers = 6,
                dim_feedforward = 2048, 
                dropout = 0.2,
                activation = "relu",
                batch_first = False,
                )
        

        self.fc1 = nn.Linear(self.hid_size, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        #x = x.permute(1,0,2)    # [L, N, Hin]
        print(x.shape)
        src = self.l1(x.permute(1,0,2)[:,:,-1])
        print(src.shape)
        src = self.f1(src)
        src = src.unsqueeze(0)
        print(src.shape)
        emb = torch.cat([x, src], 2)
        print(emb.shape)
        fesfsef

        h0 = torch.randn(2, 1, self.hid_size + 2).double()
        c0 = torch.randn(2, 1, self.hid_size + 2).double()
        # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x, (h_n, h_c) = self.trans(x, (h0, c0))
        x = F.relu(x)
        x = self.fc1(x[-1,:,:])
        x = F.dropout(x, 0.2)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, 0.2)
        #x = torch.squeeze(x, 0)
        #print(x.shape)
        #esfs
        return x

class L_05(nn.Module):
    def __init__(self, input_size):
        super(L_05, self).__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = 48,
                batch_first = False,
                num_layers = 1,
                dropout = 0,
                )
        self.fc1 = nn.Linear(48, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        # print(x.shape)
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.squeeze(x, 0)
        x = torch.squeeze(x, 0)

        #print(x.shape)
        return x


from torch.autograd import Variable
class LM(nn.Module):
    def __init__(self, num_classes=1, input_size=4, hidden_size=28, num_layers=1):
        super(LM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 8
        
        self.first = nn.Sequential(
            nn.Conv1d(input_size, hidden_size*2, 3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size*2, input_size, 3, dilation=2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.last = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, num_classes),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        #print(x.shape)
        #x = x.permute(0,2,1)    # [L, N, Hin]
        #print(x.shape)
        #x = self.first(x)
        #x = x.permute(0,2,1)
        #print(x.shape)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).double())
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).double())
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        out = self.last(h_out)
        #print(out.shape)
        return out

class L_05(nn.Module):
    def __init__(self, input_size):
        super(L_05, self).__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = 48,
                batch_first = False,
                num_layers = 1,
                dropout = 0,
                )
        self.fc1 = nn.Linear(48, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        # print(x.shape)
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.squeeze(x, 0)
        x = torch.squeeze(x, 0)

        #print(x.shape)
        return x

class L_05_B(nn.Module):
    def __init__(self, input_size):
        super(L_05_B, self).__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = 48,
                batch_first = False,
                num_layers = 2,
                dropout = 0,
                bidirectional = True
                )
        self.fc1 = nn.Linear(96, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.squeeze(x, 0)
        x = torch.squeeze(x, 0)

        #print(x.shape)
        return x

class CGAN_G(nn.Module):
    def __init__(self, input_size):
        super(CGAN_G, self).__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = 48,
                batch_first = False,
                num_layers = 2,
                dropout = 0,
                bidirectional = True
                )
        self.fc1 = nn.Linear(96, 108)
        self.fc2 = nn.Linear(108, 108)
        self.fc3 = nn.Linear(108, 1)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x, _ = self.lstm(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.squeeze(x, 0)
        x = torch.squeeze(x, 0)
        return x

class CGAN_D(nn.Module):
    def __init__(self, input_size):
        super(CGAN_D, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(1, 8, (1, 1))
        self.conv2 = nn.Conv2d(8, 16, (1, 1))
        self.conv3 = nn.Conv2d(16, 32, (1, 1))

        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        #print(x.shape)
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #print(x.shape)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc5(x)
        x = torch.sigmoid(x)
        x = torch.squeeze(x, 0)
        x = torch.squeeze(x, 0)
        return x