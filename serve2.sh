source /mnt/shared-storage-user/mineru4s/dingruiyi/anaconda/bin/activate
conda activate prm
cd /mnt/shared-storage-user/mineru4s/dingruiyi/srpo
gunicorn -k gevent -w 8 --worker-connections 1000 -b 0.0.0.0:4997 wsgi:app
