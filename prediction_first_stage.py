import numpy as np
import torch.nn as nn
import os, shutil, torch
import matplotlib.pyplot as plt
from utils.config import opt
from load_data import IMG_Folder
from model import ScaleDense
from model import CNN
from model import ResNet
from model import VGG
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from load_mask_data import IMG_Class_Folder
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
import torch.nn.functional as F
import cv2
from load_data import nii_loader
from sklearn.metrics import mean_absolute_error, r2_score


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


def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()
    pred = pred.data.numpy()
    mae = mean_absolute_error(target, pred)
    return mae


def main():
    # ======== define data loader and CUDA device ======== #
    test_data = IMG_Folder(opt.excel_path, opt.test_folder)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ========  build and set model  ======== #
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
        model = EfNetB0.from_name("efficientnet-b0",
                                  override_params={
                                      'num_classes': 1,
                                      'dropout_rate': 0.2
                                  },
                                  in_channels=1,
                                  )
    elif opt.model == 'VIT':
        model = VisionTransformer(num_layers=2)
    elif opt.model == 'Multi_VIT':
        model = MultiViewViT(
            image_sizes=[(91, 109), (91, 91), (109, 91)],
            patch_sizes=[(7, 7), (7, 7), (7, 7)],
            num_channals=[91, 109, 91],
            vit_args={'emb_dim': 768, 'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 12, 'num_classes': 1,
                      'dropout_rate': 0.1, 'attn_dropout_rate': 0.0},
            mlp_dims=[3, 128, 256, 512, 1024, 512, 256, 128, 1]
        )
    elif opt.model == "MultiViewViT_SelfAttention":
        model = MultiViewViT_SelfAttention(image_sizes=[(91, 109), (91, 91), (109, 91)],
                                           patch_sizes=[(7, 7), (7, 7), (7, 7)],
                                           num_channals=[91, 109, 91],
                                           vit_args={'emb_dim': 768, 'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 10,
                                                     'num_classes': 768},
                                           mlp_dims=[768, 512, 256, 1]
                                           )
    elif opt.model == "MultiViewViT_CBAM":
        model = MultiViewViT_ConvAttention(image_sizes=[(91, 109), (91, 91), (109, 91)],
                                           patch_sizes=[(7, 7), (7, 7), (7, 7)],
                                           num_channals=[91, 109, 91],
                                           vit_args={'emb_dim': 768, 'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 10,
                                                     'num_classes': 768},
                                           mlp_dims=[768, 512, 256, 1]
                                           )
    elif opt.model == 'Multi_ResNet':
        model = MultiViewResNet(
            mlp_dims=[1536, 512, 256, 1]
        )
    elif opt.model == 'Multi_CNN':
        model = MultiViewCNN(
            mlp_dims=[768, 512, 256, 1]
        )
    elif opt.model == 'Multi-VGG':
        model = MultiViewVGG(mlp_dims=[12288, 4096, 1024, 256, 1])
    else:
        print('Wrong model choose')

    # ======== load trained parameters ======== #
    model = nn.DataParallel(model).to(device)
    criterion = nn.MSELoss().to(device)
    model.load_state_dict(torch.load(os.path.join(opt.output_dir + opt.model + '_best_model.pth.tar'))['state_dict'])

    # ======== build data loader ======== #
    test_loader = torch.utils.data.DataLoader(test_data
                                              , batch_size=opt.batch_size
                                              , num_workers=opt.num_workers
                                              , pin_memory=True
                                              , drop_last=True
                                              )

    # ======== test preformance ======== #
    test(valid_loader=test_loader
         , model=model
         , criterion=criterion
         , device=device
         , save_npy=True
         , npy_name=opt.npz_name
         , figure=True
         , figure_name='../training_loss/' + opt.model + ' True_age_and_predicted_age.png')


def visulizaton_VIT(att_mat, im):
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(device="cuda:0")
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device="cuda:0")
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[0, 1:].to(device="cpu")
    # grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # 改attention方向时修改这里，需要与7除
    mask = v.reshape(15, 13).detach().numpy()
    mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[0]))
    result = (mask * im).astype("uint8")
    return result


def find_most_frequent_element(tensor):
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    _, max_idx = torch.max(counts, dim=0)
    most_frequent = unique_elements[max_idx]
    return most_frequent.item()


def test(valid_loader, model, criterion, device
         , save_npy=False, npy_name='test_result.npz'
         , figure=False, figure_name='../training_loss/' + opt.model + ' True_age_and_predicted_age.png'):
    '''
    [Do Test process according pretrained model]

    Args:
        valid_loader (torch.dataloader): [test set dataloader defined in 'main']
        model (torch CNN model): [pre-trained CNN model, which is used for brain age estimation]
        criterion (torch loss): [loss function defined in 'main']
        device (torch device): [GPU]
        save_npy (bool, optional): [If choose to save predicted brain age in npy format]. Defaults to False.
        npy_name (str, optional): [If choose to save predicted brain age, what is the npy filename]. Defaults to 'test_result.npz'.
        figure (bool, optional): [If choose to plot and save scatter plot of predicted brain age]. Defaults to False.
        figure_name (str, optional): [If choose to save predicted brain age scatter plot, what is the png filename]. Defaults to 'True_age_and_predicted_age.png'.

    Returns:
        [float]: MAE and pearson correlation coeficent of predicted brain age in teset set.
    '''

    losses = AverageMeter()
    MAE = AverageMeter()

    model.eval()  # switch to evaluate mode
    out, targ, ID, Attn1, Attn2, Attn3 = [], [], [], [], [], []
    target_numpy, predicted_numpy, ID_numpy = [], [], []
    print('======= start prediction =============')
    # ======= start test programmer ============= #
    with torch.no_grad():

        for _, (input, ids, target, male) in enumerate(valid_loader):
            input = input.to(device).type(torch.FloatTensor)

            # ======= convert male lable to one hot type ======= #
            male = torch.unsqueeze(male, 1)
            male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1)
            male = male.type(torch.FloatTensor).to(device)

            target = torch.from_numpy(np.expand_dims(target, axis=1))
            target = target.type(torch.FloatTensor).to(device)

            # ======= compute output and loss ======= #
            if opt.model == 'ScaleDense':
                output = model(input, male)

            else:
                output = model(input)
                #output, (attn1, attn2, attn3) = model(input, return_attention_weights=True)
            out.append(output.cpu().numpy())
            targ.append(target.cpu().numpy())
            ID.append(ids)
            # attn1= torch.stack(attn1, dim=0)
            # attn2 = torch.stack(attn2, dim=0)
            # attn3 = torch.stack(attn3, dim=0)
            # attn1_mean = attn1.mean(dim=0)
            # attn2_mean = attn2.mean(dim=0)
            # attn3_mean = attn3.mean(dim=0)
            # avg_attn1 = attn1_mean.mean(dim=1)
            # avg_attn2 = attn2_mean.mean(dim=1)
            # avg_attn3 = attn3_mean.mean(dim=1)
            # avg_attn1 = avg_attn1.mean(dim=0)
            # avg_attn2 = avg_attn2.mean(dim=0)
            # avg_attn3 = avg_attn3.mean(dim=0)
            # Attn1.append(avg_attn1)
            # Attn2.append(avg_attn2)
            # Attn3.append(avg_attn3)

            loss = criterion(output, target)
            mae = metric(output.detach(), target.detach().cpu())

            # ======= measure accuracy and record loss ======= #
            losses.update(loss, input.size(0))
            MAE.update(mae, input.size(0))

        targ = np.asarray(targ)
        out = np.asarray(out)
        ID = np.asarray(ID)

        for idx in targ:
            for i in idx:
                target_numpy.append(i)

        for idx in out:
            for i in idx:
                predicted_numpy.append(i)

        for idx in ID:
            for i in idx:
                ID_numpy.append(i)

        target_numpy = np.asarray(target_numpy)
        predicted_numpy = np.asarray(predicted_numpy)
        ID_numpy = np.asarray(ID_numpy)

        errors = predicted_numpy - target_numpy
        abs_errors = np.abs(errors)
        errors = np.squeeze(errors, axis=1)
        abs_errors = np.squeeze(abs_errors, axis=1)
        target_numpy = np.squeeze(target_numpy, axis=1)
        predicted_numpy = np.squeeze(predicted_numpy, axis=1)
        # Attn1 = torch.mean(torch.stack(Attn1), dim=0)
        # Attn2 = torch.mean(torch.stack(Attn2), dim=0)
        # Attn3 = torch.mean(torch.stack(Attn3), dim=0)

        # original_data = nii_loader("../data/70s/sub-1004-nonlin_brain.nii.gz")
        # 修改attention map方向时，修改下面的代码
        # original_data = original_data[45, :, :]

        # attn_map = visulizaton_VIT(Attn1, original_data)
        # torch.save(attn_map, '../attn_map/0s/attn_map1.pt')

        # attn_map2= visulizaton_VIT(Attn2,original_data)
        # torch.save(attn_map2, '../attn_map/80s/attn_map2.pt')

        # Attn3 = Attn3[:-1, :-1]
        # attn_map3 = visulizaton_VIT(Attn3, original_data)

        # torch.save(attn_map3, '../attn_map/70s/attn_map3.pt')

        r2 = r2_score(target_numpy, predicted_numpy)

        # ======= output several results  ======= #
        print('===============================================================\n')
        print(
            'TEST  : [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f}, R²: {r2:.4f} \n'.format(
                len(valid_loader), loss=losses, MAE=MAE, r2=r2))

        print('STD_err = ', np.std(errors))
        print(' CC:    ', np.corrcoef(target_numpy, predicted_numpy))
        print('PAD spear man cc', spearmanr(errors, target_numpy, axis=1))
        print('spear man cc', spearmanr(predicted_numpy, target_numpy, axis=1))
        print('mean pad:', np.mean(errors))

        print('\n =================================================================')

        if save_npy:
            savepath = os.path.join(opt.output_dir, npy_name)
            np.savez(savepath
                     , target=target_numpy
                     , prediction=predicted_numpy
                     , ID=ID_numpy)

        # ======= Draw scatter plot of predicted age against true age ======= #
        if figure is True:
            plt.figure()
            lx = np.arange(np.min(target_numpy), np.max(target_numpy))
            plt.plot(lx, lx, color='red', linestyle='--')
            plt.scatter(target_numpy, predicted_numpy)
            plt.xlabel('Chronological Age')
            plt.ylabel('predicted brain age')
            # plt.savefig('D:/TSAN-brain-age-estimation-master/画图/pre_vs_act.png')
            # plt.show()

        return MAE, np.corrcoef(target_numpy, predicted_numpy)


if __name__ == "__main__":
    main()