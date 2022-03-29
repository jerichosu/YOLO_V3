"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import sys
sys.path.append('.')

import config
import torch
import torch.optim as optim
# from torch.utils.data import DataLoader
# import albumentations as A
# import cv2
# from albumentations.pytorch import ToTensorV2


from model import YOLOv3
# from dataset import YOLODataset
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss

print('all modules imported')

# import warnings
# warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    model.train()
    loop = tqdm(train_loader, ncols=80)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        
    # for batch_idx, (x, y) in enumerate(train_loader):
        
        
        # x = x.to(config.DEVICE)
        # y0, y1, y2 = (
        #     y[0].to(config.DEVICE),
        #     y[1].to(config.DEVICE),
        #     y[2].to(config.DEVICE),
        # )
        
        x = x.to('cuda')
        y0, y1, y2 = (
            y[0].to('cuda'),
            y[1].to('cuda'),
            y[2].to('cuda'),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        # print('neam loss: ', mean_loss)
        loop.set_postfix(loss=mean_loss) #show loss within [] as well



def main():
    
    model = YOLOv3(num_classes=20).to("cuda")
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # config.DATASET = 'PASCAL_VOC'
    # train_csv_path = 'PASCAL_VOC/train.csv', test_csv_path = 'PASCAL_VOC/test.csv'
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )
    


    # this works
    for batch_idx, (x, y) in enumerate(train_loader):
        print(batch_idx)
        print(x.shape)
        print(y[0].shape)
        print(y[1].shape)
        print(y[2].shape)
        break
    print('Data loaded')
    
    
    
    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    #     )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    
    print('training...')
    for epoch in range(config.NUM_EPOCHS):
        print('Train Epoch: {}/{}'.format(epoch+1, config.NUM_EPOCHS))

        
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader = train_loader,
                  model = model,
                  optimizer = optimizer,
                  loss_fn = loss_fn,
                  scaler = scaler,
                  scaled_anchors = scaled_anchors)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 1 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            # mapval = mean_average_precision(
            #     pred_boxes,
            #     true_boxes,
            #     iou_threshold=config.MAP_IOU_THRESH,
            #     box_format="midpoint",
            #     num_classes=config.NUM_CLASSES,
            # )
            # print(f"MAP: {mapval.item()}")
            # model.train()


if __name__ == "__main__":
    main()
