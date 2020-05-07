import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms) 

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 315

    train_dataset = EpicDataset4BBOX(get_transform(train=True))

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=1,
            collate_fn=utils.collate_fn
            )

    model = get_detect_bbox_model(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
    print(model)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

