import numpy as np
import torch.nn as nn
import os, torch
from utils.config import opt
from load_data import IMG_Folder, nii_loader
from model import ScaleDense
from model import CNN
from model import ResNet
from model import VGG
from model import GlobalLocalTransformer
from model.efficientnet_pytorch_3d import EfficientNet3D as EfNetB0
from model.vit import VisionTransformer
from model.MultiViewViT import MultiViewViT
from model.MultiViewResNet import MultiViewResNet
from model.MultiViewCNN import MultiViewCNN
from model.MultiViewVGG import MultiViewVGG
from nilearn import datasets
import nibabel as nib
from scipy.ndimage import zoom
import pandas as pd


def main():
    # ======== define data loader and CUDA device ======== #
    test_data = IMG_Folder(opt.excel_path, opt.mask_folder)
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
    model.load_state_dict(
        torch.load(os.path.join(opt.output_dir + opt.model + '_best_model.pth.tar'))['state_dict'])
    model.eval()

    # ======== build data loader ======== #
    test_loader = torch.utils.data.DataLoader(test_data
                                              , batch_size=1
                                              , num_workers=opt.num_workers
                                              , pin_memory=True
                                              , drop_last=True
                                              )

    # Get sample image to determine exact shape
    for sample_data in test_loader:
        sample_img = sample_data[0]
        img_shape = (sample_img.shape[1], sample_img.shape[2], sample_img.shape[3])
        print(f"Detected image shape: {img_shape}")
        break

    # ======== Load AAL atlas ======== #
    aal_atlas = datasets.fetch_atlas_aal()
    atlas_filename = aal_atlas.maps
    atlas_nii = nib.load(atlas_filename)
    atlas_data = atlas_nii.get_fdata()
    region_labels = np.unique(atlas_data)[1:]  # Exclude 0 (background)
    region_mapping = {code: label for code, label in zip(region_labels, aal_atlas.labels)}

    print(f"Number of regions in atlas: {len(region_labels)}")
    print(f"Atlas shape: {atlas_data.shape}")

    # ======== perform region-based occlusion analysis ======== #
    region_occlusion_results = perform_region_occlusion_analysis(
        test_loader=test_loader,
        model=model,
        device=device,
        atlas_data=atlas_data,
        region_labels=region_labels,
        region_mapping=region_mapping,
        img_shape=img_shape
    )

    # Save occlusion map and region results
    save_dir = '../Gender/'
    os.makedirs(save_dir, exist_ok=True)

    # Save as both numpy array and CSV for easy analysis
    np.save(f'{save_dir}{opt.model}_region_occlusion_results.npy', region_occlusion_results)

    # Convert to dataframe and save as CSV
    df = pd.DataFrame({
        'region_id': region_labels,
        'region_name': [region_mapping[r] for r in region_labels],
        'occlusion_effect': region_occlusion_results
    })
    df.to_csv(f'{save_dir}{opt.model}_region_occlusion_results.csv', index=False)

    # Also save a full 3D occlusion map for visualization
    occlusion_map_3d = region_to_voxel_map(region_occlusion_results, atlas_data, region_labels)
    np.save(f'{save_dir}{opt.model}_occlusion_map_3d.npy', occlusion_map_3d)

    # Save as NIfTI file for visualization in neuroimaging tools
    save_nifti(occlusion_map_3d, f'{save_dir}{opt.model}_occlusion.nii.gz', atlas_nii.affine)


def resample_to_target_shape(data, target_shape):
    """
    Resample data to match the target shape
    """
    # Calculate zoom factors
    factors = (target_shape[0] / data.shape[0],
               target_shape[1] / data.shape[1],
               target_shape[2] / data.shape[2])

    # Resample using order=1 (linear interpolation) for continuous data
    resampled_data = zoom(data, factors, order=1)

    return resampled_data


def perform_region_occlusion_analysis(test_loader, model, device, atlas_data, region_labels, region_mapping, img_shape):
    """
    Perform occlusion analysis based on brain regions defined in the atlas
    """
    model.eval()

    print(f"Target image shape for resampling: {img_shape}")

    # Check if the atlas needs resampling
    if atlas_data.shape != img_shape:
        print(f"Resampling atlas from {atlas_data.shape} to {img_shape}")
        resampled_atlas = resample_to_target_shape(atlas_data, img_shape)
    else:
        print("Atlas already matches target shape, no resampling needed")
        resampled_atlas = atlas_data

    print(f"Resampled atlas shape: {resampled_atlas.shape}")

    # Initialize results dictionary to store effect per region
    region_occlusion_effects = {region: 0 for region in region_labels}
    region_sample_counts = {region: 0 for region in region_labels}

    print('======= Starting Region-Based Occlusion Analysis =============')

    sample_count = 0

    with torch.no_grad():
        for _, (input_img, ids, target, male) in enumerate(test_loader):
            # Print input_img shape for debugging
            print(f"Input image shape: {input_img.shape}")

            # Debugging: Print detailed shape information
            print(f"Input data type: {input_img.dtype}")

            # Get original prediction
            input_img = input_img.to(device).type(torch.FloatTensor)

            # Handle gender information if needed by model
            if opt.model == 'ScaleDense':
                male_onehot = torch.unsqueeze(male, 1)
                male_onehot = torch.zeros(male_onehot.shape[0], 2).scatter_(1, male_onehot, 1)
                male_onehot = male_onehot.type(torch.FloatTensor).to(device)
                original_output = model(input_img, male_onehot)
            else:
                original_output = model(input_img)

            original_output = original_output.cpu().numpy()

            # Process each region one by one
            for region in region_labels:
                # Free up memory
                torch.cuda.empty_cache()

                # Create mask for this region
                region_mask = (resampled_atlas == region)

                # Skip if region is not present in the resampled atlas
                if not np.any(region_mask):
                    continue

                # Clone the original input
                masked_input = input_img.clone()

                # Move input to CPU for masking
                cpu_input = masked_input.cpu().numpy()

                # Create a zero array with the same shape
                zeroed_array = np.zeros_like(cpu_input)

                # Create a mask array by broadcasting the region mask
                # This safely handles all dimension arrangements
                mask_array = np.ones_like(cpu_input)

                # Apply the region mask - this is the key change
                # We're assuming the last 3 dimensions of cpu_input correspond to the 3D volume
                for i in range(cpu_input.shape[0]):  # batch dimension
                    # Create a view that can be applied to the 3D volume regardless of channel arrangement
                    mask_view = np.broadcast_to(~region_mask, cpu_input[i].shape)
                    cpu_input[i] = cpu_input[i] * mask_view

                # Move back to GPU
                masked_input = torch.from_numpy(cpu_input).to(device)

                # Get prediction for masked input
                if opt.model == 'ScaleDense':
                    masked_output = model(masked_input, male_onehot)
                else:
                    masked_output = model(masked_input)

                masked_output = masked_output.cpu().numpy()

                # Calculate effect for this region (difference from original)
                effect = masked_output - original_output

                # Accumulate effect for this region
                region_occlusion_effects[region] += effect.item()
                region_sample_counts[region] += 1

                # Clean up
                del masked_input, cpu_input
                if 'masked_output' in locals():
                    del masked_output
                torch.cuda.empty_cache()

            sample_count += 1
            print(f"Processed sample {sample_count}/{len(test_loader)}: {ids[0]}")

    # Average effects across samples
    for region in region_labels:
        if region_sample_counts[region] > 0:
            region_occlusion_effects[region] /= region_sample_counts[region]

    # Convert results to a structured array
    result_array = np.array([region_occlusion_effects[region] for region in region_labels])

    return result_array


def region_to_voxel_map(region_effects, atlas_data, region_labels):
    """
    Convert region-based effects to a voxel-wise map for visualization
    """
    # Initialize 3D map with zeros
    voxel_map = np.zeros_like(atlas_data, dtype=np.float32)

    # Fill in values based on region effects
    for i, region in enumerate(region_labels):
        voxel_map[atlas_data == region] = region_effects[i]

    return voxel_map


def save_nifti(data, filename, affine=None):
    """
    Save data as a NIfTI file
    """
    if affine is None:
        # Default affine if none provided
        affine = np.eye(4)

    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filename)


if __name__ == "__main__":
    main()

