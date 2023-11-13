from torch import cat
import torch.nn as nn

class PinnNet(nn.Module):

    def __init__(self, num_features, num_outputs, num_con_input=0):
        super(PinnNet, self).__init__()
        self.lin1 = nn.Linear(num_features, 512)
        self.lin2 = nn.Linear(512, 256)
        
        self.lin22 = nn.Linear(256, 256)
        
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, num_outputs)
        self.activation = nn.Tanh()
        # self.activation = nn.SiLU()
        # self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        
        x = self.lin22(x)
        x = self.activation(x)
        
        x = self.lin3(x)
        x = self.activation(x)
        x = self.lin4(x)
        x = self.activation(x)
        x = self.lin5(x)
        x = self.activation(x)
        x = self.lin6(x)
        return x

    
class LargePinnNet(nn.Module):

    def __init__(self, num_features, num_outputs, num_con_input=0):
        super(LargePinnNet, self).__init__()
        self.lin1 = nn.Linear(num_features, 728)
        self.lin2 = nn.Linear(728, 512)
        
        self.lin22 = nn.Linear(512, 256)
        self.lin23 = nn.Linear(256, 256)
        
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, num_outputs)
        self.activation = nn.Tanh()
        # self.activation = nn.SiLU()
        # self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        
        x = self.lin22(x)
        x = self.activation(x)
        x = self.lin23(x)
        x = self.activation(x)

        x = self.lin3(x)
        x = self.activation(x)
        x = self.lin4(x)
        x = self.activation(x)
        x = self.lin5(x)
        x = self.activation(x)
        x = self.lin6(x)
        return x
    
    

class PinnNetConcat(nn.Module):

    def __init__(self, num_features, num_outputs, num_con_input=30):
        super(PinnNetConcat, self).__init__()
        self.lin1 = nn.Linear(num_features, 512)
        self.lin2 = nn.Linear(512+num_con_input, 256)
        self.lin3 = nn.Linear(256+num_con_input, 128)
        self.lin4 = nn.Linear(128+num_con_input, 64)
        self.lin5 = nn.Linear(64+num_con_input, 32)
        self.lin6 = nn.Linear(32+num_con_input, 16)
        self.lin7 = nn.Linear(16+num_con_input, 8)
        self.lin8 = nn.Linear(8+num_con_input, num_outputs)
        
        self.activation = nn.Tanh()
        self.num_con_input = num_con_input
        # self.activation = nn.SiLU()
        # self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xin):
        x = self.lin1(xin)
        x = self.activation(x)
        x = cat([xin[:, :self.num_con_input], x], dim=1)

        x = self.lin2(x)
        x = self.activation(x)
        x = cat([xin[:, :self.num_con_input], x], dim=1)
        
        x = self.lin3(x)
        x = self.activation(x)
        x = cat([xin[:, :self.num_con_input], x], dim=1)
        
        x = self.lin4(x)
        x = self.activation(x)
        x = cat([xin[:, :self.num_con_input], x], dim=1)
        
        x = self.lin5(x)
        x = self.activation(x)
        x = cat([xin[:, :self.num_con_input], x], dim=1)
        
        x = self.lin6(x)
        x = self.activation(x)
        x = cat([xin[:, :self.num_con_input], x], dim=1)
        
        x = self.lin7(x)
        x = self.activation(x)
        x = cat([xin[:, :self.num_con_input], x], dim=1)
        
        x = self.activation(x)
        x = self.lin8(x)
        return x
    
    
class SmallPinnNet(nn.Module):

    def __init__(self, num_features, num_outputs, num_con_input=0):
        super(SmallPinnNet, self).__init__()
        self.lin1 = nn.Linear(num_features, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_outputs)
        self.activation = nn.Tanh()
        # self.activation = nn.SiLU()
        # self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        x = self.lin3(x)
        return x

class ExtraLargePinnNet(nn.Module):

    def __init__(self, num_features, num_outputs, num_con_input=0):
        super(ExtraLargePinnNet, self).__init__()
        self.lin1 = nn.Linear(num_features, 1024)
        self.lin2 = nn.Linear(1024, 728)
        
        self.lin22 = nn.Linear(728, 640)
        self.lin23 = nn.Linear(640, 512)
        self.lin24 = nn.Linear(512, 256)
        
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, num_outputs)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        
        x = self.lin22(x)
        x = self.activation(x)
        x = self.lin23(x)
        x = self.activation(x)
        x = self.lin24(x)
        x = self.activation(x)

        x = self.lin3(x)
        x = self.activation(x)
        x = self.lin4(x)
        x = self.activation(x)
        x = self.lin5(x)
        x = self.activation(x)
        x = self.lin6(x)
        return x