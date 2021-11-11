CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --world_size=2 \
--dataset_folder your_train_dataset_folder_path \
--csv_file_path your_annoatation_file_path \
--num_class 100 \
--output_file_path output