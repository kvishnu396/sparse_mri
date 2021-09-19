import numpy as np
from scipy.signal import convolve2d

# smoothed coil estimation
def smoothed_coil_estimation(dmri_img):
    nx,ny,nc,nt = dmri_img.shape
    
    # initial estimate
    img_combined = np.sqrt(np.sum(np.abs(dmri_img)**2,2))
    img_combined = np.expand_dims(img_combined,axis=2)
    S_0 = dmri_img/img_combined
    
    # use property of smoothness 
    # convolve previous estimate with smoothing kernel
    S_1 = np.empty((nx,ny,nc,nt),dtype=S_0.dtype)
    kernel = np.ones((9,9),dtype=S_0.dtype)/9**2
    for c in range(nc):
        for t in range(nt):
            S_1[:,:,c,t] = convolve2d(S_0[:,:,c,t],kernel,mode='same')
    
    # mask sensitivies
    thresh = 0.05*np.max(np.abs(img_combined))
    mask = np.repeat(img_combined,nc,axis=2) > thresh
    S_2 = S_1*mask
    #print(S_2.shape,len(np.where(S_2[:,:,0,0]>0)[0]))
    
    return S_2


# Adaptive sensitivity estimation
# described in Adaptive reconstruction of phased array MR imagery
# need to finish
def adaptive_sens_est(dmri_img):
    nx,ny,nc,nt = dmri_img.shape
    
    S = np.zeros((nx,ny,nc,nt))
    M = np.zeros((nx,ny,nc,nt))
    w = 5
    for t in range(nt):
        for x in range(nx):
            dim_x = [e for e in range(max(x-w,1),min(x+w,nx)+1)]
            for y in range(ny):
                dim_y = [e for e in range(max(y-w,1),min(y+w,ny)+1)]
                kernel = np.reshape(dmri_img[dim_x,dim_y,:,t],(-1,nc))
    

