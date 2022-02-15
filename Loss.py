
import tensorflow as tf
from tensorflow import keras
from keras import Input,Model

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet',input_tensor=Input(shape=(128, 128, 3)))
inter_vgg=[]

for i in range(8):
    inter_vgg.append(Model(inputs = vgg.input, outputs=vgg.get_layer(vgg.layers[i].name).output))

class loss_function(keras.losses.Loss):
    def __init__(self,w_mse,w_ssim,w_perloss,w_fftloss):
        super(loss_function,self).__init__()
        self.w_mse = w_mse
        self.w_ssim = w_ssim
        self.w_perloss = w_perloss
        self.w_fftloss = w_fftloss
    def nmse_loss(self,pred,gt):
        mse = tf.keras.metrics.mean_squared_error(pred,gt)
        mse = tf.math.reduce_sum(mse,axis=(1,2))
        norm = tf.norm(gt,axis=(1,2))
        norm = tf.squeeze(norm)
        norm = tf.pow(norm,2)
        norm = tf.math.reduce_sum(norm)
        nmse = tf.math.divide(mse,norm)
        nmse = tf.math.reduce_mean(nmse)
        return nmse
    def fft_loss(self,pred,gt):
        pred = tf.transpose(pred, perm=[0, 3, 1, 2])
        gt   = tf.transpose(gt, perm=[0, 3, 1, 2])

        pred_fft = tf.signal.fftshift(tf.signal.rfft2d(pred))
        gt_fft   = tf.signal.fftshift(tf.signal.rfft2d(gt))

        pred_fft = tf.transpose(pred_fft, perm=[0, 2, 3, 1])
        gt_fft   = tf.transpose(gt_fft, perm=[0, 2, 3, 1])

        ft_loss = self.nmse_loss(pred_fft,gt_fft)
        ft_loss = tf.cast(ft_loss,tf.float32)
        return ft_loss

    def ssim_loss(self,pred,gt):
        sim_loss = 1.0-tf.math.reduce_mean(tf.image.ssim(pred,gt,max_val=1))
        return sim_loss

    def percep_loss(self,pred,gt):
        ploss = 0
        pred = tf.image.grayscale_to_rgb(pred)
        gt   = tf.image.grayscale_to_rgb(gt)
        for i in range(8):
            vgg_pred = inter_vgg[i](pred)
            vgg_gt = inter_vgg[i](gt)
            ploss = ploss+self.nmse_loss(vgg_pred,vgg_gt)
        return ploss
    
    def loss_value(self,pred,gt):
        L1 = self.nmse_loss(pred,gt)
        L2 = self.fft_loss(pred,gt)
        L3 = self.ssim_loss(pred,gt)
        L4 = self.percep_loss(pred,gt,inter_vgg)
        loss = self.w_mse*L1 + self.w_fftloss*L2 + self.w_ssim*L3 + self.w_perloss*L4
        return loss



