import numpy as np 
import pydicom
import matplotlib.pyplot as plt

def normalize_data_ab(a, b, data):
    # input (min_data, max_data) with range (max_data - min_data) is normalized to (a, b)
    min_x = min(data.ravel())
    max_x = max(data.ravel())  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def normalize_data_ab_cd(a, b, c, d, data):
    # input data (min_data, max_data) with range (d-c) is normalized to (a, b)
    min_x = c
    max_x = d  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def add_rnl_white(rnl, b, mu, sigma):
    """ sigma = std 
    var = sigma^2
    """
    h, w = b.shape
    randn =  np.random.normal(loc = mu, scale = sigma, size = (h,w))
    e = randn/np.linalg.norm(randn, ord = 2)
    e = rnl*np.linalg.norm(b)*e;
    return(b + e)

def pydicom_imread(path):
  """ reads dicom image with filename path 
  and dtype be its original form
  """
  input_image = pydicom.dcmread(path)
  return(input_image.pixel_array.astype('float32'))

def plot2dlayers(arr, xlabel=None, ylabel=None, title=None, cmap=None, colorbar=True):
    """
    'brg' is the best colormap for reb-green-blue image
    'brg_r': in 'brg' colormap green color area will have
        high values whereas in 'brg_r' blue area will have
        the highest values
    """
    if xlabel is None:
        xlabel=''
    if ylabel is None:
        ylabel=''
    if title is None:
        title=''
    if cmap is None:
        cmap='Greys_r'
    plt.imshow(arr, cmap=cmap)
    cb = plt.colorbar()
    if colorbar is False:
      cb.remove()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def multi2dplots(nrows, ncols, fig_arr, axis, passed_fig_att=None):
    """
      gf.multi2dplots(1, 2, lena_stack, axis=0, passed_fig_att={'colorbar': False, 'split_title': np.asanyarray(['a','b']),'out_path': 'last_lr.tif'})
      where lena_stack is of size (2, 512, 512)
    """
    default_att= {"suptitle": '',
            "split_title": np.asanyarray(['']*(nrows*ncols)),
            "supfontsize": 12,
            "xaxis_vis"  : False,
            "yaxis_vis"  : False,
            "out_path"   : '',
            "figsize"    : [8, 8],
            "cmap"       : 'Greys_r',
            "plt_tight"  : True,
            "colorbar"   : True
                 }
    if passed_fig_att is None:
        fig_att = default_att
    else:
        fig_att = default_att
        for key, val in passed_fig_att.items():
            fig_att[key]=val
    
    f, axarr = plt.subplots(nrows, ncols, figsize = fig_att["figsize"])
    img_ind  = 0
    f.suptitle(fig_att["suptitle"], fontsize = fig_att["supfontsize"])
    for i in range(nrows):
        for j in range(ncols):                
            if (axis==0):
                each_img = fig_arr[img_ind, :, :]
            if (axis==1):
                each_img = fig_arr[:, img_ind, :]
            if (axis==2):
                each_img = fig_arr[:, :, img_ind]
                
            if(nrows==1):
                ax = axarr[j]
            elif(ncols ==1):
                ax =axarr[i]
            else:
                ax = axarr[i,j]
            im = ax.imshow(each_img, cmap = fig_att["cmap"])
            if fig_att["colorbar"] is True:  f.colorbar(im, ax=ax)
            ax.set_title(fig_att["split_title"][img_ind])
            ax.get_xaxis().set_visible(fig_att["xaxis_vis"])
            ax.get_yaxis().set_visible(fig_att["yaxis_vis"])
            img_ind = img_ind + 1
            if fig_att["plt_tight"] is True: plt.tight_layout()
            
    if (len(fig_att["out_path"])==0):
        plt.show()
    else:
        plt.savefig(fig_att["out_path"])

def raw_imread(path, shape=(256, 256), dtype='int16'):
  input_image = np.fromfile(path, dtype=dtype).astype('float32')
  input_image = input_image.reshape(shape)
  return(input_image)

def dict_plot_of_2d_arr(rows, cols, arr_2d, cmap='Greys_r', save_plot=False, disp_plot=False, output_path=''):
  # rows, cols indicate number of subplots along rows & columns
  # rows*cols = len(arr_2d)
  plt.figure(figsize=(14, 10))
  for i, comp in enumerate(arr_2d):
      plt.subplot(rows, cols, i + 1)
      plt.imshow(comp, cmap=cmap, interpolation="nearest")
      plt.xticks(())
      plt.yticks(())
  plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
  if save_plot: plt.savefig(output_path)
  if disp_plot: plt.show()