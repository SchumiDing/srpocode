
rlaunch --gpu=1 --memory=64000 --cpu=4 \
    --charged-group=mineru4sh_gpu --private-machine=yes \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    -- bash -c "python testmodel.py"