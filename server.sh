rlaunch --gpu=2 --memory=16000 --cpu=16 \
    --charged-group=mineru4sh_gpu --private-machine=yes \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    -- bash -c "python3 /mnt/shared-storage-user/mineru4s/dingruiyi/srpo/rmserver.py"
