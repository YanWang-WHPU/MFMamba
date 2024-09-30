import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
from fvcore.nn import flop_count 
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils_Mamba import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.MFMamba import MFMamba, load_pretrained_ckpt

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

if MODEL == 'MFMamba':
    net = MFMamba(num_classes=N_CLASSES).cuda()
    net = load_pretrained_ckpt(net)

flops_dict =[]
input = torch.randn(32, 3, 256, 256).cuda()  
input2 = torch.randn(32, 256, 256).cuda()  
flops = flop_count(net, (input,input2))  
flops_dict = flops[0]
print(f"FLOPs: {flops_dict}") 
# for op, flops_value in flops_dict.items():  
#     print(f"  {op}: {flops_value:.2f} GFLOPs") 
total_flops = sum(flops_dict.values()) 
print(f"Total FLOPs in GigaFLOPs: {total_flops / 10:.2f}")




params = 0
for name, param in net.named_parameters(): 
    params += param.nelement()
print(params)


# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# Load the datasets
print("training : ", str(len(train_ids)) + ", testing : ", str(len(test_ids)) + ", Stride_Size : ", str(Stride_Size), ", BATCH_SIZE : ", str(BATCH_SIZE))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)
print("Train dataset:", train_loader) 
print("Length of train_loader:", len(train_loader)) 
if train_loader is None:  
    raise ValueError("Train dataset is None. Please check your dataset initialization.")


train_loss = []
train_acc = []


base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1) #
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 25, 35], gamma=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.1)

################### 
def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    if DATASET == 'Urban':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids) 
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids) 
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64') - 1) for id in test_ids)
    elif DATASET == 'Vaihingen':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids) 
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids) 
        test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids) 
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    elif DATASET == 'Potsdam' :
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids) 
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids) 
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms,test_labels, eroded_labels), total=len(test_ids), leave=False): 
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)): 
                #### Build the tensor
                ## rgb
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                
                ## dsm 
                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)           

                # Do the inference
                outs = net(image_patches ,dsm_patches)
                # outs = net(image_patches ,image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array 
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()

       
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


######################## 
def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    weights = weights.cuda() 


    iter_ = 0
    MIoU_best = 0.82
    for e in range(1, epochs + 1): 
        if scheduler is not None: 
            scheduler.step()
        net.train() 
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()),  Variable(target.cuda()) 
            optimizer.zero_grad() 

            output = net(data,dsm) 
            # output = net(data,data) # RGB
            # output = net(dsm,dsm)


            loss = loss_calc(output, target, weights) 

            loss.backward()
            optimizer.step() 

            # losses[iter_] = loss.data
            # mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])
   
            train_loss.append(loss.item())


            if iter_ % 100 == 0:  
                clear_output() 
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0) 
                gt = target.data.cpu().numpy()[0] 
                accuracy_value = accuracy(pred, gt)
                train_acc.append(accuracy_value)
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLr: {:.6f}\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], loss.data, accuracy(pred, gt)))
            iter_ += 1 

            del (data, target, loss) 

            ## !!! You can increase the frequency of testing to find better models
            if iter_ % 1000 == 0: 
                # We validate with the largest possible stride for faster computing
                net.eval() 
                MIoU = test(net, test_ids, all=False, stride=Stride_Size)
                net.train() 
                if MIoU > MIoU_best: 
                    if DATASET == 'Vaihingen':
                        torch.save(net.state_dict(), './resultsv/Vaihingen/duibi_global/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                    elif DATASET == 'Urban':
                        torch.save(net.state_dict(), './resultsv/Urban/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                    elif DATASET == 'Potsdam':
                        torch.save(net.state_dict(), './resultsv/Potsdam/RGB/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                    MIoU_best = MIoU  
        
        ## Figure loss,acc
        plt.figure(figsize=(10, 5)) 
        plt.plot(train_loss, label='Train Loss')  
        plt.title('Train_Loss Curve')  
        plt.xlabel('Epoch')  
        plt.ylabel('Loss')  
        plt.grid(True)  
        plt.savefig(os.path.join('./resultsv/Potsdam/RGB', 'loss.png'))
        plt.show()   

        plt.figure(figsize=(10, 5))  
        plt.plot(train_acc, label='Train Accuracy')  
        plt.title('Train_acc Curve')  
        plt.xlabel('Epoch')  
        plt.ylabel('acc')  
        plt.grid(True)  
        plt.savefig(os.path.join('./resultsv/Potsdam/RGB', 'acc.png'))
        plt.show()         

if MODE == 'Train': 
    time_start=time.time()
    train(net, optimizer, 50, scheduler) 
    time_end=time.time()
    print('Total Time Cost: ',time_end-time_start)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/Vaihingen/duibi_global/RS3Mamba_epoch10_0.826030680496352'), strict=False) 
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32) 
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p) 
            # plt.imshow(img) and plt.show()
            io.imsave('./resultsv/Vaihingen/duibi_global/inference8260_'+MODEL+'_tile_{}.png'.format(id_), img) 

    elif DATASET == 'Urban':
        net.load_state_dict(torch.load('./resultsu/UNetformer_Embed_epoch23_0.5055703729081706'), strict=False)  
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./resultsu/inference5058_'+MODEL+'_tile_{}.png'.format(id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsv/Potsdam/RGB/RS3Mamba_epoch29_0.850354972268778'), strict=False) 
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32) 
        print("MIoU: ", MIoU) 
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./resultsv/Potsdam/RGB/inference8503_'+MODEL+'_tile_{}.png'.format(id_), img) 