NGPUS=1
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS test.py \
    --name cifar100-retrain --dataset cifar100 --model_type cifar \
    --n_classes 100 --init_channels 36 --stem_multiplier 3 \
    --batch_size 128 --workers 1 --print_freq 100 \
    --cell_file 'cells/cifar_genotype.json' \
    --world_size $NGPUS --distributed  --dist_url 'tcp://127.0.0.1:24443' \
    --resume --resume_name 'c100.pth.tar'
