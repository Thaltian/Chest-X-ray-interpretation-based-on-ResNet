import torch
from torch import optim, nn
import visdom
from utils import Flatten
from torch.utils.data import DataLoader
import os
from xray_dataset_pretreat import xray
import matplotlib.pyplot as plt
import time
from torchvision.models import resnet18

#record loss and accuracy
lossline = []
accline = []
train_accline=[]
lossline_x = []
accline_x = []

#Hyperparmeter
batchsz = 150
lr = 0.2e-4
epochs = 5
device = torch.device('cuda:0')
torch.manual_seed(1234)

#data pretreat
train_db = xray('train', 224, mode='train')
val_db = xray('eval', 224, mode='val')
test_db = xray('test', 224, mode='test')
root = 'test'
parent_path = os.path.dirname(root)
mdl_save_path = os.path.join(parent_path, 'best_module', 'best_transfer_mouth.mdl')
realmdl_save_path = os.path.join(parent_path, 'best_module', 'realbest_transfer_mouth.mdl')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=6)
val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=True, num_workers=6)
test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=True, num_workers=6)

viz = visdom.Visdom()

#evalute accuracy in evaluation dataset or test dataset.
def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        a, b = x, y
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            c = pred.cpu()
        correct += torch.eq(pred, y).sum().float().item()

    viz.images(train_db.denormalize(a), nrow=8, win='batch', opts=dict(title='batch'))
    viz.text(str(b.numpy()), win='label1', opts=dict(title='batch-y'))
    viz.text(str(c.numpy()), win='label2', opts=dict(title='batch-pred'))
    return correct / total


def main():
    #We use pretrained model by Imagenet
    trained_model = resnet18(pretrained=True)
    print(trained_model)
    #We need to change the last output layer
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          nn.Dropout(p=0.5),
                          nn.Linear(512, 15)
                          ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    global_step2 = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    val_acc = evalute(model, val_loader)
    train_acc = evalute(model, train_loader)
    print('init_train_acc:', train_acc)
    print('init_val_acc:', val_acc)
    
    train_accline.append(train_acc)
    accline.append(val_acc)
    accline_x.append(global_step2)
    viz.line([val_acc], [global_step2], win='val_acc', update='append')
    #start training
    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossline.append(loss.item())
            lossline_x.append(global_step + 1)
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:
            global_step2 += 1
            print('AI have finished:', epoch + 1, 'epochs!')
            train_acc = evalute(model, train_loader)
            print('time',time.time())
            val_acc = evalute(model, val_loader)
            print('time', time.time())


            train_accline.append(train_acc)
            print('val_acc:', val_acc)
            print('train_acc:', train_acc)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), mdl_save_path)
                torch.save(model, realmdl_save_path)

            accline.append(val_acc)
            accline_x.append(global_step2)
            viz.line([val_acc], [global_step2], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch + 1)

    model.load_state_dict(torch.load(mdl_save_path))

    print('AI get ready to use best module to test testset!')
    print('AI have loaded module successfully!')

    test_acc = evalute(model, test_loader)
    print('AI get the test acc:', test_acc)



    plt.figure(num=1, figsize=(15, 5))
    # print(accline)
    # print(accline_x)
    # print(lossline)

    plt.plot(lossline_x, lossline, color='red')
    plt.title('training loss')
    plt.xlabel('batch')
    plt.ylabel('loss')

    plt.figure(num=2)


    l1,=plt.plot(accline_x, accline, color='green')
    l2,=plt.plot(accline_x, train_accline, color='red')
    plt.legend(handles=[l2, l1], labels=["train_acc", "eval_acc"], loc="best", fontsize=6)
    plt.title('accuracy in evaluation dataset')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    print('done!')

    plt.show()

if __name__ == '__main__':
    main()
