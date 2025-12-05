source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate
conda activate verl
path=$1
output=$2
echo "merging $path to $output"
python /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/merge_to_hf.py \
    --checkpoint_dir $path \
    --output_dir /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/$output
echo "merging done"