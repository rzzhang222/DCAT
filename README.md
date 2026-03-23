# DCAT
code for paper Semi-supervised Adversarial Training via Disentangled Contrastive Learning
To get the results, run the following code:
CUDA_VISIBLE_DEVICES=0 python3 train_doubleaugbetaregscheduleablation60.py configs/double_cifar100.yaml --seed 1 --model wrn-28-10

This code is based on the code of WSCAT: https://github.com/zhang-lilin/WSCAT. Thanks for their excellent work.
