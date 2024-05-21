# Getting started with OccFormer

For most of our experiments, we train the model with 8 RTX A40 GPUs with 46G or 8 RTX A800 GPUs with 80G memory. Therefore, you may need similar hardwares to reproduce the training results.

Before start training, download the corresponding pretrained backbones from the [release page](https://github.com/Rorisis/Co-Occ/releases/tag/pretrained) and put them under the folder `ckpts/`. The weights include [R50](https://github.com/Rorisis/Co-Occ/releases/tag/pretrained/resnet50-0676ba61.pth) & [R101-DCN](https://github.com/Rorisis/Co-Occ/releases/tag/pretrained/resnet101-5d3b4d8f.pth) for nuScenes.

## Training
```bash
bash tools/dist_train.sh $CONFIG 8
```
During the training process, the model is evaluated on the validation set after every epoch. The checkpoint with best performance will be saved. The output logs and checkpoints will be available at work_dirs/$CONFIG.

## Evaluation
Evaluate with 1 GPU:
```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval=bbox
```
The single-GPU inference will print the current performance after each iteration, which can serve as a quick indicator.

Evaluate with 8 GPUs:
```bash
bash tools/dist_test.sh $YOUR_CONFIG $YOUR_CKPT 8
```

