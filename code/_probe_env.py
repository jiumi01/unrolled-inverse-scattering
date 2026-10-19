import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
try:
    import torch.func as tf
    print('torch.func.jvp:', hasattr(tf, 'jvp'))
except Exception as e:
    print('torch.func fail:', e)
try:
    import scipy
    from scipy.sparse.linalg import svds, LinearOperator
    print('scipy', scipy.__version__, 'svds ok')
except Exception as e:
    print('scipy fail:', e)
