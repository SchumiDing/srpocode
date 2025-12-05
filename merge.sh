model_path=$1

rlaunch --gpu=1 --memory=96000 --cpu=16 \
    --charged-group=mineru4sh_gpu --private-machine=yes \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    -- bash -c "python model_merger.py merge --backend fsdp --local_dir ${model_path}; python testmodel.py"