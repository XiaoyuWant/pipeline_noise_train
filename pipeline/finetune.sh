CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 finetune.py --world_size=2 \
--csv_file_path output/annos_of_1340.csv \
--num_class 1340 \
--model efficientnet \
--batchsize 64
