
# change this variable to match your machine.
N_GPU=8

# JPEG restoration
python train.py --n-gpu-per-node $N_GPU --beta-max 0.3 --corrupt jpeg-5
python train.py --n-gpu-per-node $N_GPU --beta-max 0.3 --corrupt jpeg-10

# 4x super-resolution
python train.py --n-gpu-per-node $N_GPU --beta-max 0.3 --corrupt sr4x-pool
python train.py --n-gpu-per-node $N_GPU --beta-max 0.3 --corrupt sr4x-bicubic

# Deblurring
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt blur-uni
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt blur-gauss

# Inpainting
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt inpaint-center
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt inpaint-freeform1020
python train.py --n-gpu-per-node $N_GPU --beta-max 1.0 --ot-ode --corrupt inpaint-freeform2030
