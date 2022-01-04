#%%
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *

class twist(nn.Module):
    def __init__(self):
        super(twist, self).__init__()
        self.revolute = nn.Parameter(torch.Tensor(6)).uniform_(-1,1)
        self.prismatic = nn.Parameter(torch.Tensor(6)).uniform_(-1,1)
 
        self.position = nn.Parameter(torch.Tensor(1,3).uniform_(-1,1)) 
        self.orientation = nn.Parameter(torch.Tensor(1,3).uniform_(-1,1))
    
    def forward(self,out, q_single_value):
        revoluteTwist = inv_x(t2x(out)) @ self.revolute
        out = out @ srodrigues(self.revolute, q_single_value[:,0])

        prismaticTwist =  inv_x(t2x(out)) @ self.prismatic
        out = out @ srodrigues(self.prismatic, q_single_value[:,1])

        return revoluteTwist,prismaticTwist,out
        
class POELayer(nn.Module):
    def __init__(self, branchLs):
        super(POELayer, self).__init__()

        self.branchLS = branchLs
        n_joint = len(branchLs)

        for joint in range(n_joint):
            setattr(self,'Twist_'+str(joint), twist())


    def forward(self, q_value):
        branchLs = self.branchLS
        n_joint = len(branchLs)
        batch_size = q_value.size()[0]
        device = q_value.device
        out = torch.tile(torch.eye(4),(batch_size,1,1)).to(device)
        revoluteTwistls = torch.zeros([batch_size,n_joint,6]).to(device)
        prismaticTwistls = torch.zeros([batch_size,n_joint,6]).to(device)
        

        outs = torch.tensor([]).reshape(batch_size,-1,4,4).to(device)
        for joint in range(n_joint):
            Twist = getattr(self, 'Twist_'+str(joint))
            q_single_value = q_value[:,joint,:]
            revoluteTwist,prismaticTwist,out_temp = Twist(q_single_value)

            revoluteTwistls[:,joint,:] = revoluteTwist
            prismaticTwistls[:,joint,:] = prismaticTwist
            out = out_temp
        
            if branchLs[joint]:
                p = Twist.postion
                rpy = Twist.orientation
                r = rpy2r(rpy)
                out_temp = out @ pr2t(p, r)
                outs = torch.cat((outs,out_temp.unsqueeze(1)), dim=1)

        return outs,revoluteTwistls,prismaticTwistls

class q_layer(nn.Module):
    def __init__(self,branchLs,inputdim,n_layers=7):
        super(q_layer, self).__init__()
        n_joint = len(branchLs)
        
        LayerList = []
        for _ in range(n_layers):
            layer = nn.Linear(inputdim,2*inputdim)
            torch.nn.init.xavier_uniform_(layer.weight)
            LayerList.append(layer)
            inputdim = inputdim * 2

        for _ in range(n_layers-3):
            layer = nn.Linear(inputdim,inputdim//2)
            torch.nn.init.xavier_uniform_(layer.weight)
            LayerList.append(layer)
            inputdim = inputdim // 2

        layer = nn.Linear(inputdim,n_joint)
        torch.nn.init.xavier_uniform_(layer.weight)
        LayerList.append(layer)

        self.layers = torch.nn.ModuleList(LayerList)
        

    def forward(self, motor_control):
        out =motor_control
        
        for layer in self.layers:
            out = layer(out)
            out = torch.nn.LeakyReLU()(out)
    
        q_value = out
        return q_value

class Model(nn.Module):
    def __init__(self, branchLs, inputdim):
        super(Model,self).__init__()
        self.q_layer = q_layer(branchLs, inputdim)
        self.poe_layer = POELayer(branchLs)

    def forward(self, motor_control):
        out = self.q_layer(motor_control)
        SE3,_ = self.poe_layer(out)

        return SE3




#%%