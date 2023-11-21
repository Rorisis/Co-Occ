import torch
file_path = './work_dirs/cascade_voxel_nusc_multi_ray/best_SSC_mean_epoch_6.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)

# smaller 63374123
# v4 69140395
