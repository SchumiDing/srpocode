# check if path and output are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: testrj <checkpoint_path> <dataset_path>"
    exit 1
fi

path=$2
dataset_path=$1


name1="${path##*/}"
name2="${dataset_path##*/}"
name1="${name1//./}"
name2="${name2//./}"
name="$name1-$name2-tester"
rjob delete $name
rjob submit \
    --name=$name  \
    --gpu=1 \
    --memory=320000 \
    --cpu=16 \
    --charged-group=mineru4sh_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --image=registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/yehc:torch-2.6.0-57d787c2-0627 \
    --host-network=true \
    -- bash -c "bash /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/test.sh $path $dataset_path ${3:-all}"
