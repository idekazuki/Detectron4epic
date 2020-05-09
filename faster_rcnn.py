import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.multiprocessing 
torch.multiprocessing.set_sharing_strategy('file_system')

from epicdatasets import  EpicDataset4BBOX
from engine import train_one_epoch
import utils
import transforms as T

def get_detect_bbox_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    hidden_layer = 256
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
    #    transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms) 

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 352

    train_dataset = EpicDataset4BBOX(get_transform(train=True))

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn
            )

    model = get_detect_bbox_model(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
    print(model)
    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
        save_log_path = './logfile/save_{}.pth'.format(epoch)       
        status = {       
            'epoch_num': epoch + 1,       
            'state_dict': model.state_dict(),       
            'optimizer': optimizer.state_dict(),       
        }       
        torch.save(status, save_log_path) 
        lr_scheduler.step()

