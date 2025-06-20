# Check if CUDA is available
train_on_gpu  = torch.cuda.is_available()

if not train_on_gpu :
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
