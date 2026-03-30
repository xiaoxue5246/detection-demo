import torch
import engine
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

from engine import get_data_loader, Averager, train, validate
from model import model

# Matplotlib, Python画图工具包
matplotlib.style.use('ggplot')

# 定义使用的计算资源类型：GPU和CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 主代码传参
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--show-sample', dest='show_sample', default='no', 
                 help='whether to visualize a wheat sample with bboxes or not')
args = vars(parser.parse_args())

# 学习参数配置
num_epochs = 20
lr = 0.001
batch_size = 4

model = model().to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None

# loss计算初始化
loss_hist = engine.Averager()

# 【数据加载】 加载训练和验证数据
train_data_loader, valid_data_loader = get_data_loader(batch_size)

# if args['show_sample'] == 'yes':
#     images, targets, image_ids = next(iter(train_data_loader))
#     images = list(image.to(device) for image in images)
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#     boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
#     sample = images[2].permute(1,2,0).cpu().numpy()
#     fig, ax = plt.subplots(1, 1, figsize=(16, 8))
# 
#     for box in boxes:
#         cv2.rectangle(sample,
#                       (box[0], box[1]),
#                       (box[2], box[3]),
#                       (220, 0, 0), 3)
#     
#     ax.set_axis_off()
#     ax.imshow(sample)
#     plt.show()

iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

# 【训练和验证】主程序
train_loss = []
precision = []
for epoch in range(num_epochs):
    itr = 1
    train_loss_hist, end, start = train(train_data_loader, lr_scheduler,
                                        model, optimizer, device,
                                        epoch, loss_hist, itr)
    valid_prec = validate(valid_data_loader, model, device, iou_thresholds)
    print(f"Took {(end-start)/60:.3f} minutes for epoch# {epoch} to train")
    print(f"Epoch #{epoch} Train loss: {train_loss_hist.value}")  
    print(f"Epoch #{epoch} Validation Precision: {valid_prec}")  
    train_loss.append(train_loss_hist.value)
    precision.append(valid_prec)
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_own.pth')

# plot and save the training loss
plt.figure()
plt.plot(train_loss, label='Training loss')
plt.legend()
plt.show()
plt.savefig('loss.png')

# plot and save the validation precision
plt.figure()
plt.plot(precision, label='Validation precision')
plt.legend()
plt.show()
plt.savefig('precision.png')