import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torchvision.transforms.functional import scale
from model import Model
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------超参数---------------------------------
batch_size = 8
learning_rate = 0.000025
epochs = 20

#--------------------载入数据---------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dir = './train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

val_dir = './val'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
print(train_datasets.class_to_idx) #查看子文件夹与标签的映射，注意：不是顺序映射



#--------------------载入模型---------------------------------
model = Model(2)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

#--------------------训练过程---------------------------------
for epoch in range(epochs):
    model.train()
    # print('Start Train')
    train_loss = 0
    train_acc = 0
    epoch_size =  len(train_datasets)//batch_size
    with tqdm(total=  epoch_size,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataloader):
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images  = torch.Tensor(images).cuda()
                targets = targets.cuda()
            out = model(images)
            loss = loss_func(out, targets)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == targets).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'train_loss': train_loss / (iteration+1), 
                                'train_acc'        : train_acc / (iteration*batch_size +1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
   
    model.eval()
    # print('Start Validation')
    val_loss = 0
    val_acc = 0
    epoch_size =  len(val_datasets)//batch_size
    with tqdm(total=  epoch_size,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_dataloader):
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images  = torch.Tensor(images).cuda()
                targets = targets.cuda()
            
            optimizer.zero_grad()
            out = model(images)
            loss = loss_func(out, targets)
            val_loss += loss.item()
            pred = torch.max(out, 1)[1]
            val_correct = (pred == targets).sum()
            val_acc += val_correct.item()

            pbar.set_postfix(**{'val_loss': val_loss / (iteration+1), 
                                'val_acc'        : val_acc / (iteration*batch_size +1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    lr_scheduler.step()
    torch.save(model.state_dict(), 'best.pth')
    # torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f.pth'%((epoch+1),train_loss/(epoch_size+1)))


