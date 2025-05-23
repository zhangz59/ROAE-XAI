import os
import torch
import json
import datetime
import warnings
import tensorboardX
import numpy as np
import torch.nn as nn
from utils.config import opt
from model import ScaleDense
from model import CNN
from model import ResNet
from model import VGG
from model.ranking_loss import rank_difference_loss
from load_data import IMG_Folder
from prediction_first_stage import test
from sklearn.metrics import mean_absolute_error
from model import GlobalLocalTransformer
from model import vgg_4_trans
from model.efficientnet_pytorch_3d import EfficientNet3D as EfNetB0
from model.vit import VisionTransformer
from model.MultiViewViT import MultiViewViT
from model.MultiViewResNet import MultiViewResNet
from model.MultiViewCNN import MultiViewCNN
from model.MultiViewVGG import MultiViewVGG
from model.MultiViewViT_SelfAttention import MultiViewViT_SelfAttention
from model.MultiViewViT_ConvAttention import MultiViewViT_ConvAttention

import timm



print(torch.cuda.is_available())

torch.cuda.current_device()
torch.cuda._initialized = True





warnings.filterwarnings("ignore")

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.enabled = True




 # ======== main function =========== #
print('===== hyper-parameter ====== ')
print("=> network     : {}".format(opt.model))
print("=> lambda      : {}".format(opt.lbd))
print("=> batch size  : {}".format(opt.batch_size))
print("=> learning rate    : {}".format(opt.lr))
print("=> weight decay     : {}".format(opt.weight_decay))
print("=> aux loss         : {}x{}".format(opt.aux_loss, opt.lbd))

def main(res):
    best_metric = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    json_path = os.path.join(opt.output_dir,'hyperparameter.json')
    with open(json_path,'w') as f:
        f.write(json.dumps(vars(opt)
                            ,ensure_ascii=False
                            ,indent=4
                            ,separators=(',', ':')))
    print("=========== start train the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))


    # ===========  define data folder and CUDA device =========== #
    train_data = IMG_Folder( opt.excel_path
                            ,opt.train_folder)
    valid_data = IMG_Folder( opt.excel_path
                            ,opt.valid_folder)
    

    # ===========  define data loader =========== #
    train_loader = torch.utils.data.DataLoader(  train_data
                                                ,batch_size=opt.batch_size
                                                ,num_workers=opt.num_workers
                                                ,pin_memory=True
                                                ,drop_last=True
                                                ,shuffle=True
                                                )
    valid_loader = torch.utils.data.DataLoader(  valid_data
                                                ,batch_size=opt.batch_size 
                                                ,num_workers=opt.num_workers 
                                                ,pin_memory=True
                                                ,drop_last=True
                                                )
    
    # ===========  build and set model  =========== #  
    if opt.model == 'ScaleDense':
        model = ScaleDense.ScaleDense(8, 5, opt.use_gender)
    elif opt.model == 'CNN':
        model = CNN.CNNModel()
    elif opt.model == 'resnet':
        model = ResNet.resnet50()
    elif opt.model == 'VGG':
        model = VGG.vgg()
    elif opt.model == 'Transformer':
        model = GlobalLocalTransformer.GlobalLocalBrainAge(1,
                        patch_size=64,
                        step=32,
                        nblock=6,
                        backbone='vgg16')
    elif opt.model == 'EfficientNet':
        model=EfNetB0.from_name("efficientnet-b0",
                              override_params={
                                  'num_classes': 1,
                                  'dropout_rate': 0.2
                              },
                              in_channels=1,
                              )
    elif opt.model == 'VIT':
        model= VisionTransformer(num_layers=2)
    elif opt.model == 'Multi_VIT':
        model = MultiViewViT(
            image_sizes=[(91, 109), (91, 91), (109, 91)],
            patch_sizes=[(7, 7), (7, 7), (7, 7)],
            num_channals=[91,109,91],
            vit_args={'emb_dim': 768, 'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 12, 'num_classes': 1,
                      'dropout_rate': 0.1, 'attn_dropout_rate': 0.0},
            mlp_dims=[3,128,256,512,1024,512,256,128,1]

        )
    elif opt.model=="MultiViewViT_SelfAttention":
        model = MultiViewViT_SelfAttention(image_sizes=[(91, 109), (91, 91), (109, 91)],
            patch_sizes=[(7, 7), (7, 7), (7, 7)],
            num_channals=[91,109,91],
            vit_args={'emb_dim': 768,'mlp_dim': 3072,'num_heads': 12,'num_layers': 10,'num_classes': 768},
            mlp_dims=[768, 512, 256, 1]
                                  )
    elif opt.model=="MultiViewViT_CBAM":
        model = MultiViewViT_ConvAttention(image_sizes=[(91, 109), (91, 91), (109, 91)],
            patch_sizes=[(7, 7), (7, 7), (7, 7)],
            num_channals=[91,109,91],
            vit_args={'emb_dim': 768,'mlp_dim': 3072,'num_heads': 12,'num_layers': 10,'num_classes': 768},
            mlp_dims=[768, 512, 256, 1]
                                  )
    elif opt.model == 'Multi_ResNet':
        model= MultiViewResNet(
            mlp_dims=[1536, 512, 256, 1]
        )
    elif opt.model == 'Multi_CNN':
        model=MultiViewCNN(
            mlp_dims=[768, 512, 256, 1]
        )
    elif opt.model == 'Multi_VGG':
        model= MultiViewVGG(mlp_dims=[12288, 4096, 1024, 256, 1])

    else:
        print('Wrong model choose')

    model.apply(weights_init)
    model = nn.DataParallel(model).to(device)
    model_test = model

    # =========== define the loss function =========== #
    loss_func_dict = {'mae': nn.L1Loss().to(device)
                     ,'mse': nn.MSELoss().to(device)
                     ,'ranking':rank_difference_loss(sorter_checkpoint_path=opt.sorter
                                                    ,beta=opt.beta).to(device)
                     }
        
    criterion1 = loss_func_dict[opt.loss]
    criterion2 = loss_func_dict[opt.aux_loss]

    # =========== define optimizer and learning rate scheduler =========== #
    optimizer = torch.optim.Adam( model.parameters()
                                 ,lr=opt.lr
                                 ,weight_decay=opt.weight_decay
                                 ,amsgrad=True
                                )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer
                                                           ,verbose=1
                                                           ,patience=5
                                                           ,factor=0.5
                                                           )
    early_stopping = EarlyStopping(patience=200, verbose=True)
    
    # =========== define tensorboardX and show traing start signal =========== #
    saved_metrics, saved_epos = [], []
    num_epochs = int(opt.epochs)
    sum_writer = tensorboardX.SummaryWriter(opt.output_dir)
    print(" ==========>  is getting started...")
    print(" ==========> Training takes {} epochs.".format(num_epochs))

    # =========== start train =========== #
    for epoch in range(opt.epochs):
            
        # ===========  train for one epoch   =========== #
        train_loss, train_mae = train(  train_loader=train_loader
                                      , model=model
                                      , criterion1=criterion1

                                      , optimizer=optimizer
                                      , device=device
                                      , epoch=epoch)

        # ===========  evaluate on validation set ===========  #
        valid_loss, valid_mae = validate( valid_loader=valid_loader
                                        , model=model 
                                        , criterion1=criterion1

                                        , device=device)        

        # ===========  learning rate decay =========== #  
        scheduler.step(valid_mae)
        for param_group in optimizer.param_groups:
            print("\n*learning rate {:.2e}*\n" .format(param_group['lr']))

        # ===========  write in tensorboard scaler =========== #
        sum_writer.add_scalar('train/loss', train_loss, epoch)
        sum_writer.add_scalar('train/mae', train_mae, epoch)
        sum_writer.add_scalar('valid/loss', valid_loss, epoch)
        sum_writer.add_scalar('valid/mae', valid_mae, epoch)

        # ===========  record the  best metric and save checkpoint ===========  #
        valid_metric = valid_mae
        is_best = False
        if valid_metric < best_metric:
            is_best = True
            best_metric = min(valid_metric, best_metric)
                
            saved_metrics.append(valid_metric)
            saved_epos.append(epoch)
            print('=======>   Best at epoch %d, valid MAE %f\n' % (epoch, best_metric))

        save_checkpoint({'epoch': epoch
                        ,'arch': opt.model
                        ,'state_dict': model.state_dict()
                        ,'optimizer': optimizer.state_dict()}
                        , is_best
                        , opt.output_dir
                        , model_name=opt.model
                        )

        # ===========  early_stopping needs the validation loss or MAE to check if it has decresed 
        early_stopping(valid_mae)
        if early_stopping.early_stop:
            print("======= Early stopping =======")
            break

    # =========== write traning and validation log =========== #
    os.system('echo " ================================== "')
    os.system('echo " ==== TRAIN MAE mtc:{:.5f}" >> {}'.format(train_mae, res))
    
    print('Epo - Mtc')
    mtc_epo = dict(zip(saved_metrics, saved_epos))
    rank_mtc = sorted(mtc_epo.keys(), reverse=False)
    try:
        for i in range(10):
            print('{:03} {:.3f}'.format(mtc_epo[rank_mtc[i]]
                                       ,rank_mtc[i]))
            os.system('echo "epo:{:03} mtc:{:.3f}" >> {}'.format(
                                                                  mtc_epo[rank_mtc[i]]
                                                                , rank_mtc[i]
                                                                , res))
    except:
        pass
    
    # ===========  clean up ===========  #
    torch.cuda.empty_cache()
    # =========== test the trained model on test dataset =========== #
    test_data = IMG_Folder(opt.excel_path, opt.test_folder)
    test_loader = torch.utils.data.DataLoader( test_data
                                              ,batch_size=opt.batch_size 
                                              ,num_workers=opt.num_workers 
                                              ,pin_memory=True
                                              ,drop_last=True)
    
    # =========== test on the best model on test data =========== # 
    model_best = model_test
    model_best.load_state_dict( torch.load(
                                os.path.join(opt.output_dir+opt.model+'_best_model.pth.tar')
                                )['state_dict']) 
    print('========= best model test result ===========')
    test_MAE,test_CC = test( test_loader
                            ,model_best
                            ,criterion1
                            ,device
                            ,npy_name='best_model_test_result'
                            ,save_npy=True
                            ,figure=False
                            ,figure_name='best_model.png')
    os.system('echo " ================================== "')
    os.system('echo "best valid model TEST MAE mtc:{:.5f}" >> {}'.format(test_MAE.avg, res))
    os.system('echo "best valid model TEST rr mtc:{:.5f}" >> {}'.format(test_CC[0][1], res))

train_loss = []

def train(train_loader, model, criterion1,  optimizer, device, epoch):
    '''
    For training process

    Args:
        train_loader (data loader): train data loader.
        model (CNN model): convolutional neural network.
        criterion1 (loss fucntion): main loss function.
        criterion2 (loss fucntion): aux loss function.
        optimizer (torch.optimizer): optimizer which is defined in 'main'
        device (torch device type): default: GPU
        epoch (int): training epoch idex

    Returns:
        [float]: training loss average and MAE average
    '''
    
    losses = AverageMeter()
    MAE = AverageMeter()
    #LOSS1 = AverageMeter()


    for i, (img,_,target, male) in enumerate(train_loader):
        target = torch.from_numpy(np.expand_dims(target,axis=1))

        # =========== convert male lable to one hot type =========== #
        if opt.use_gender:
            male = torch.unsqueeze(male,1)
            male = torch.zeros(male.shape[0],2).scatter_(1,male,1)
            male = male.to(device).type(torch.FloatTensor)
        input = img.to(device)
        target = target.type(torch.FloatTensor).to(device)
        # target = torch.squeeze(target, dim=1)

        # =========== compute output and loss =========== #
        model.train()
        model.zero_grad()
        if opt.model == 'ScaleDense':
            out = model(input, male)

        else:
            #out = model(input)
            out,(attn1, attn2, attn3) = model(input,return_attention_weights=True)


        # =========== compute loss =========== #
        loss = criterion1(out, target)
        #if opt.lbd > 0:
            #loss2 = criterion2(out, target)
        #else:
            #loss2 = 0
        #loss = loss1 + opt.lbd * loss2
        #loss=loss1

        mae = metric(
                out.detach(), target.detach().cpu())
        losses.update(loss, input.size(0))
        #LOSS1.update(loss1,input.size(0))
        #LOSS2.update(loss2,img.size(0))
        MAE.update(mae, input.size(0))
        if i % opt.print_freq == 0:
            print(
                  'Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'.format
                  ( epoch, opt.epochs, i, len(train_loader)
                  , loss=losses, MAE=MAE ))
        train_loss.append(loss.item())

        # =========== loss gradient back progation and optimizer parameter =========== #
        loss.requires_grad_(True)
        loss.backward()

        optimizer.step()
    with open("../training_loss/"+opt.model+" train_loss.txt", 'w') as train_los:
        train_los.write(str(train_loss))

    return losses.avg,MAE.avg

def validate(valid_loader, model, criterion1,  device):
    '''
    For validation process
    
    Args:
        valid_loader (data loader): validation data loader.
        model (CNN model): convolutional neural network.
        criterion1 (loss fucntion): main loss function.
        criterion2 (loss fucntion): aux loss function.
        device (torch device type): default: GPU

    Returns:
        [float]: training loss average and MAE average
    '''

    losses = AverageMeter()
    MAE = AverageMeter()

    # =========== switch to evaluate mode ===========#
    model.eval()


    with torch.no_grad():
        for _, (input,_,target,male) in enumerate(valid_loader):
            # input = input.type(torch.FloatTensor)
            input = input.to(device)

            # =========== convert male lable to one hot type =========== #
            if opt.use_gender:
                male = torch.unsqueeze(male,1)
                male = torch.zeros(male.shape[0],2).scatter_(1,male,1)
                male = male.to(device).type(torch.FloatTensor)
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = target.type(torch.FloatTensor).to(device)

            # =========== compute output and loss =========== #
            if opt.model == 'ScaleDense':
                out = model(input,male)
            else:
                #out = model(input)
                out, (attn1, attn2, attn3) = model(input, return_attention_weights=True)

            # =========== compute loss =========== #
            loss = criterion1(out, target)

            #loss = loss1
            mae = metric(out.detach(), target.detach().cpu())

            # =========== measure accuracy and record loss =========== #
            losses.update(loss, input.size(0))
            MAE.update(mae, input.size(0))
        print(
                'Valid: [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f}'.format(
                len(valid_loader), loss=losses, MAE=MAE))

        return losses.avg, MAE.avg

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

def save_checkpoint(state, is_best, out_dir, model_name):
    checkpoint_path = out_dir+model_name+'_checkpoint.pth.tar'
    best_model_path = out_dir+model_name+'_best_model.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n\n")

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(w, 'weight'):
            # nn.init.kaiming_normal_(w.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='leaky_relu')
        if hasattr(w, 'bias') and w.bias is not None:
                nn.init.constant_(w.bias, 0)
    if classname.find('Linear') != -1:
        if hasattr(w, 'weight'):
            torch.nn.init.xavier_normal_(w.weight)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)
    if classname.find('BatchNorm') != -1:
        if hasattr(w, 'weight') and w.weight is not None:
            nn.init.constant_(w.weight, 1)
        if hasattr(w, 'bias') and w.bias is not None:
            nn.init.constant_(w.bias, 0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



if __name__ == "__main__":
    res = os.path.join(opt.output_dir, 'result')
    if os.path.isdir(opt.output_dir): 
        #if input("### output_dir exists, rm? ###") == 'y':
        os.system('rm -rf {}'.format(opt.output_dir))

    # =========== set train folder =========== #
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
        
    print('=> training from scratch.\n')
    os.system('echo "train {}" >> {}'.format(datetime.datetime.now(), res))

    main(res)
