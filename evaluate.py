import torch
from model import *
from dataloader import *
from utils.pyart import *
import argparse
import numpy as np
from pathlib import Path

def main(args):
    print("Processing...")

    # make save_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    branchNum = checkpoint['branchNum']
    input_dim = checkpoint['input_dim']
    branchLs = bnum2ls(branchNum)
    n_joint = len(branchLs)

    # load model
    model = Model(branchLs, input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # load data
    test_data_loader = ToyDataloader(args.data_path, n_workers = 1, batch = 1, shuffle=False)

    # get jointAngle.txt
    jointAngle = np.array([]).reshape(-1,n_joint)
    for input,_ in test_data_loader:
        
        jointAngle_temp = model.q_layer(input)
        jointAngle_temp = jointAngle_temp.detach().cpu().numpy()
        jointAngle = np.vstack((jointAngle,jointAngle_temp))
    np.savetxt(args.save_dir+"/jointAngle.txt", jointAngle)

    # get jointTwist.txt
    jointTwist = np.array([]).reshape(-1,6)
    twists = model.poe_layer.twist
    for twist in twists:
        twist = twist.detach().cpu().numpy()
        jointTwist = np.vstack((jointTwist,twist))
    np.savetxt(args.save_dir+'/jointTwist.txt',jointTwist)

    # get branchLs
    np.savetxt(args.save_dir+'/branchLs.txt',branchLs)

    # get joint offset
    jointOffset = np.array([]).reshape(-1,6)
    for joint in range(n_joint):
        branchnameP = 'branch'+str(joint)+'_p'
        branchnameRPY = 'branch'+str(joint)+'_rpy'
        p = getattr(model.poe_layer,branchnameP).detach().cpu().numpy()[0]
        rpy = getattr(model.poe_layer,branchnameRPY).detach().cpu().numpy()[0]
        combined = np.concatenate((p,rpy))
        jointOffset = np.vstack((jointOffset,combined))
    np.savetxt(args.save_dir+'/jointOffset.txt',jointTwist)


    # get targetPose.txt
    targetPose = test_data_loader.dataset.label
    targetPose = targetPose.detach().cpu().numpy()
    np.savetxt(args.save_dir+'/targetPose.txt', targetPose)

    # get outputPose.txt
    outputPose = np.array([]).reshape(-1,targetPose.shape[1])
    for input,_ in test_data_loader:
        outputPose_temp = model(input)
        outputPose_temp = outputPose_temp[:,:,0:3,3]
        outputPose_temp = outputPose_temp.reshape(-1,outputPose_temp.size()[1]*outputPose_temp.size()[2])
        outputPose_temp = outputPose_temp.detach().cpu().numpy()[0]
        outputPose = np.vstack((outputPose,outputPose_temp))
        
    np.savetxt(args.save_dir+"/outputPose.txt", outputPose)

    print("Done...")
if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--data_path', \
        default= './data/Multi_2dim_log_spiral/fold9/Multi_2dim_log_spiral_910.txt',type=str, \
            help='path to model checkpoint')    
    args.add_argument('--checkpoint', default= './output/1230/checkpoint_40.pth',type=str,
                    help='path to model checkpoint')
    args.add_argument('--save_dir', default='./2Visualize')
    args = args.parse_args()
    main(args)