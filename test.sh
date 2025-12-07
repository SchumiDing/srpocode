source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate
conda activate verl
model_path=$1
dataset_path=$2

python /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/naivetest.py --model_path $model_path --dataset_path $dataset_path