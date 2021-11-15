CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 finetune.py --world_size=2 \
--csv_file_path your_annoatation_file_path \
--num_class 100 \
