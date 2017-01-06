# run GPU job
# retz-client run -A nushio -c 'python 01-get-sun-image.py' --stderr  --mem 16000 --gpu 1 --cpu 8 -E CUDA_PATH=/usr/local/cuda

# run CPU job
# retz-client run -A nushio -c 'python 01-get-sun-image.py' --stderr  --mem 1000 --cpu 1
# retz-client schedule -A nushio -c 'python 01-get-sun-image.py'  --mem 1000 --cpu 1
# retz-client schedule -A nushio -c 'python 04-perfect-memoization.py'  --mem 6000 --cpu 1
# retz-client run -A nushio4 -c 'python 04-perfect-memoization.py' --stderr --mem 6000 --cpu 1
retz-client schedule -A nushio4 -c 'python 04-perfect-memoization.py' --mem 6000 --cpu 1
