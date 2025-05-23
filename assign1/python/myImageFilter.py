import numpy as np

def myImageFilter(img0, h):
    """
    Applies a 2D correlation (equivalent to convolution for symmetric kernels)
    on a grayscale image using edge padding.
    
    Parameters:
      img0 (ndarray): Input 2D grayscale image.
      h (ndarray): Filter kernel (assumed to be odd-sized in both dimensions).
      
    Returns:
      img1 (ndarray): Filtered image (same shape as img0).
    """
    fh, fw = h.shape
    pad_h = fh // 2
    pad_w = fw // 2
    # Pad using edge replication
    padded = np.pad(img0, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    img1 = np.zeros_like(img0, dtype=np.float64)
    
    # Vectorized over the kernel indices (two loops only)
    for i in range(fh):
        for j in range(fw):
            img1 += h[i, j] * padded[i:i+img0.shape[0], j:j+img0.shape[1]]
            
    return img1
