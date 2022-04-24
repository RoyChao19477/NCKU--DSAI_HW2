# You can write code above the if-main block.
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import L_05_day2_C

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.

    # ------- Load PyTorch Pretrained Model ------- #
    model = L_05_day2_C(4)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0)

    checkpoint = torch.load('7d_lstm_day2_best2.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    model.double()
    # ------- end ------- #


    # ------- Load Testing Data ------- # 
    test_data = pd.read_csv(args.testing, names=['A', 'B', 'C', 'D'])
    # ------- end ------- #


    # ------- Status ------- #
    status_now = 0      # {0 : None}, {1 : Have 1}, {-1 : Short 1}
    command = []
    # ------- end ------- #


    # ------- Start predict and command ------- #
    with open(args.output, "w") as output_file:
        for day_x in range( len( test_data ) - 1 ):
            # Step 1: Predict tomorrow's stock price and the day after tomorrow's stock price
            day_value = torch.tensor( test_data.iloc[day_x].values ).unsqueeze(0).double()
            predict = model( day_value ) * torch.tensor( day_value[0][0] )
            predict = predict.squeeze(0)
            diff = predict[1] - predict[0]
            # Step 2: Check whether it will be go up/down with Step1's result
            if diff > 0:      # go up
                if status_now == 0:
                    #command.append(1)
                    output_file.write(str(1))
                    output_file.write('\n')
                    status_now = 1
                elif status_now == 1:
                    #command.append(0)
                    output_file.write(str(0))
                    output_file.write('\n')
                    status_now = 1
                elif status_now == -1:
                    #command.append(1)
                    output_file.write(str(1))
                    output_file.write('\n')
                    status_now = 0
            elif diff < 0:      # go down
                if status_now == 0:
                    #command.append(-1)
                    output_file.write(str(-1))
                    output_file.write('\n')
                    status_now = -1
                elif status_now == 1:
                    #command.append(-1)
                    output_file.write(str(-1))
                    output_file.write('\n')
                    status_now = 0
                elif status_now == -1:
                    #command.append(0)
                    output_file.write(str(0))
                    output_file.write('\n')
                    status_now = -1
            else:
                #command.append(0)
                output_file.write(str(0))
                output_file.write('\n')