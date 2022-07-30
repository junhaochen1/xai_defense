# python Version: python3.6
# Modified from the code of the paper Visualizing Deep Networks by Optimizing with Integrated Gradients
import numpy as np
import os
import tensorflow as tf
import cv2
from mytools import load_image_in_a_dir,load_results_imagespath,load_results_image, save_image,topmaxPixel,tv_norm






def Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224), Gaussian_param = [51, 50], Median_param = 11, blur_type= 'Gaussian'):
    ########################
    # Generate blurred images as the baseline
    # Parameters:
    # input_img: the original input image
    # resize_shape: the input size for the given model
    # Gaussian_param: parameters for Gaussian blur
    # blur_type: Gaussian blur or median blur or mixed blur
    ####################################################

    original_img = cv2.resize(input_img, resize_shape)
    img = np.float32(original_img)

    if blur_type =='Gaussian':   # Gaussian blur
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)
    else:
        blurred_img=None
        raise ValueError

    img_torch = tf.cast(img, tf.float32)
    img_torch = tf.expand_dims(img_torch,axis=0)

    blurred_img_torch = tf.cast(blurred_img, tf.float32)
    blurred_img_torch = tf.expand_dims(blurred_img_torch,axis=0)
    if img_torch.ndim==3:
        img_torch = tf.expand_dims(img_torch, axis=-1)
        blurred_img_torch = tf.expand_dims(blurred_img_torch, axis=-1)
    return img_torch, blurred_img_torch

def Integrated_Mask(img, blurred_img, model, category,image_name_f, max_iterations = 15, integ_iter = 20,
                    tv_beta=2, l1_coeff = 0.01*300, tv_coeff = 0.2*300, size_init = 32,ori_label=100):
    ########################
    # IGOS: using integrated gradient descent to find the smallest and smoothest area that maximally decrease the
    # output of a deep model
    # Parameters:
    # img: the original input image
    # blurred_img: the baseline for the input image
    # model: the model that you want to visualize
    # category: the classification target that you want to visualize (category=-1 means the top 1 classification label)
    # max_iterations: the max iterations for the integrated gradient descent
    # integ_iter: how many points you want to use when computing the integrated gradients
    # tv_beta: which norm you want to use for the total variation term
    # l1_coeff: parameter for the L1 norm
    # tv_coeff: parameter for the total variation term
    # size_init: the resolution of the mask that you want to generate
    ####################################################
    resize_wh = (img.shape[1], img.shape[2])

    # initialize the mask
    mask_init = np.ones((size_init, size_init), dtype=np.float32)
    mask = tf.cast(mask_init, tf.float32)
    mask = tf.expand_dims(tf.expand_dims(mask, 0),-1)
    if img.shape[-1]==3:
        mask = tf.Variable(tf.tile(mask, (1, 1, 1, 3)))
    else:
        mask = tf.Variable(mask)

    # You can choose any optimizer

    img_torch = img
    blurred_img_torch = blurred_img
    target = model(img_torch)
    category_out = tf.argmax(target,1)

    if category ==-1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)
    print("Optimizing.. ")

    curve1 = np.array([])
    curve2 = np.array([])
    curve3 = np.array([])
    curvetop = np.array([])
    curvetop_ori = np.array([])

    # Integrated gradient descent
    alpha = 0.0001
    beta = 0.2
    Img_topLS_np=0
    for i in range(max_iterations):
        print(i)
        with tf.GradientTape() as tape:
            upsampled_mask = tf.image.resize(mask,resize_wh)
            # the l1 term and the total variation term
            loss1 = l1_coeff * tf.reduce_mean(tf.abs(1 - upsampled_mask)) + \
                    tv_coeff * tv_norm(upsampled_mask, tv_beta)
            loss_all = loss1
            # compute the perturbed image
            perturbated_input_base = img_torch*upsampled_mask + \
                                     blurred_img_torch*(1 - upsampled_mask)
            for inte_i in range(integ_iter):
                integ_mask = 0.0 + ((inte_i + 1.0) / integ_iter) * upsampled_mask
                perturbated_input_integ = img_torch*integ_mask + \
                                         blurred_img_torch*(1 - integ_mask)
                # add noise
                if img_torch.shape[-1]==1:
                    noise = np.zeros((resize_wh[0], resize_wh[1], 1), dtype=np.float32)
                else:
                    noise = np.zeros((resize_wh[0], resize_wh[1], 3), dtype=np.float32)
                noise = noise + cv2.randn(noise, 0, 0.01)
                noise = tf.cast(noise, tf.float32)
                perturbated_input = perturbated_input_integ + noise
                new_image = perturbated_input
                outputs = model(new_image)
                loss2 = outputs[0, category]

                loss_all = loss_all + loss2 / integ_iter
        grads = tape.gradient(loss_all, mask)
        # compute the integrated gradients for the given target,
        # and compute the gradient for the l1 term and the total variation term
        whole_grad = grads
        loss2_ori = model(perturbated_input_base)[0, category]
        loss3 = model(perturbated_input_base)[0, ori_label]
        loss_ori = loss1 + loss2_ori # 原图的loss
        print(loss2_ori)
        if i==0:
            curve1 = np.append(curve1, loss1.numpy())
            curve2 = np.append(curve2, loss2_ori.numpy())
            curve3 = np.append(curve3, loss3.numpy())
            curvetop = np.append(curvetop, loss2_ori.numpy())
            curvetop_ori = np.append(curvetop_ori, loss3.numpy())
        loss_oridata = loss_ori.numpy()

        # LINE SEARCH with revised Armijo condition
        step = 100000.0
        MaskClone = mask
        MaskClone = MaskClone - step * whole_grad
        MaskClone = tf.clip_by_value(MaskClone, 0, 1) # clamp the value of mask in [0,1]
        mask_LS = tf.image.resize(MaskClone, resize_wh)   # Here the direction is the whole_grad
        Img_LS = img_torch*mask_LS + \
                                     blurred_img_torch*(1 - mask_LS)
        outputsLS = model(Img_LS)
        loss_LS = l1_coeff * tf.reduce_mean(tf.abs(1 - MaskClone)) + \
                  tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]
        loss_LSdata = loss_LS.numpy()
        new_condition = whole_grad ** 2  # Here the direction is the whole_grad
        new_condition = tf.reduce_sum(new_condition)
        new_condition = alpha * step * new_condition
        while loss_LSdata > loss_oridata - new_condition.numpy():
            step *= beta
            MaskClone = mask
            MaskClone = MaskClone - step * whole_grad
            MaskClone = tf.clip_by_value(MaskClone, 0, 1)
            mask_LS = tf.image.resize(MaskClone, resize_wh)
            Img_LS = img_torch * mask_LS + \
                     blurred_img_torch * (1 - mask_LS)
            outputsLS = model(Img_LS)
            loss_LS = l1_coeff * tf.reduce_mean(tf.abs(1 - MaskClone)) + \
                      tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]
            loss_LSdata = loss_LS.numpy()
            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = tf.reduce_sum(new_condition)
            new_condition = alpha * step * new_condition
            if step<0.00001:
                break
        mask = mask - step * whole_grad

        #######################################################################################################
        MaskClone = mask
        MaskClone = tf.clip_by_value(MaskClone, 0, 1)
        mask_LS = tf.image.resize(MaskClone, resize_wh)
        Img_LS = img_torch * mask_LS + \
                 blurred_img_torch * (1 - mask_LS)
        outputsLS = model(Img_LS)
        pred_top2 = tf.math.top_k(tf.squeeze(outputsLS), 2).indices
        ###########################################################
        curve1 = np.append(curve1, loss1.numpy())
        curve2 = np.append(curve2, loss2_ori.numpy())
        curve3 = np.append(curve3, loss3.numpy())
        mask = tf.Variable(tf.clip_by_value(mask, 0, 1))
        maskdata = np.array(mask)
        maskdata = np.squeeze(maskdata)
        maskdata, imgratio = topmaxPixel(maskdata, 100)
        maskdata = np.expand_dims(maskdata, axis=0)
        if maskdata.shape[1]==28:
            maskdata = np.expand_dims(maskdata, axis=-1)
        ###############################################
        Masktop = tf.cast(maskdata, tf.float32)
        # Use the mask to perturbated the input image.
        MasktopLS = tf.image.resize(Masktop, resize_wh)
        Img_topLS = tf.clip_by_value((img_torch * MasktopLS + \
                     blurred_img_torch * (1 - MasktopLS)),0,1)
        outputstopLS = model(Img_topLS)
        # loss_top1 = l1_coeff * tf.reduce_mean(tf.abs(1 - Masktop)) + \
        #             tv_coeff * tv_norm(Masktop, tv_beta)
        loss_top2 = outputstopLS[0, category]
        loss_top3 = outputstopLS[0, ori_label]
        class_output = np.array(tf.argmax(tf.squeeze(outputstopLS)))
        curvetop = np.append(curvetop, loss_top2.numpy())
        curvetop_ori = np.append(curvetop_ori, loss_top3.numpy())

        # if max_iterations >3:
        #
        #     if i == int(max_iterations / 2):
        #         if np.abs(curve2[0] - curve2[i]) <= 0.001:
        #             print('Adjust Parameter l1_coeff at iteration:', int(max_iterations / 2))
        #             l1_coeff = l1_coeff / 10
        #     elif i == int(max_iterations / 1.25):
        #         if np.abs(curve2[0] - curve2[i]) <= 0.01:
        #             print('Adjust Parameters l1_coeff again at iteration:', int(max_iterations / 1.25))
        #             l1_coeff = l1_coeff / 5

            #######################################################################################

    upsampled_mask = tf.image.resize(mask, resize_wh)

    mask = mask.numpy()

    return mask, upsampled_mask, imgratio, curvetop, curve1, curve2,curve3 ,category, curvetop_ori,class_output,pred_top2






if __name__ == '__main__':

    # model_path = 'model/resnet_20210104-011553acc81%epoch47.h5'
    # root = 'result_images/attack/miniimage/attack_deepfool_l2/episilon0.6/miniimage_ori_att_noi_image/test'
    # which_layer = "activation_15"

    # model_path = 'model/cifar10_target_modelacc_0.81.h5'
    # root = 'result_images/attack/cifar10/attack_fgsm_linf/episilon0.01/cifar10_ori_att_noi_image/test'
    # which_layer = "activation_5"

    model_path = 'model/mnist_simple0.9945.h5'
    root = 'result_images/attack/mnist/attack_pgd_linf/episilon0.02/mnist_ori_att_noi_image/test'
    which_layer = "max_pooling2d_2"
    resize_shape_ = (28, 28)
    size_init_ = 28


    model = tf.keras.models.load_model(model_path)
    model_name_f = model_path.split('/')[-1][:-3]
    train_or_test = root.split('/')[-1]
    ori_images_path, att_images_path, att_method, dataset_name, episilon_f = load_results_imagespath(root)
    resultimages_savedpath = os.path.join('result_images/gradcam_matrix/', dataset_name, model_name_f,
                                          which_layer, att_method + episilon_f, train_or_test)
    a = 24
    jiange = 100
    if not os.path.exists(resultimages_savedpath):
        os.makedirs(resultimages_savedpath)
    with open(resultimages_savedpath+"/already.txt", "a") as file:
        file.write(f"[{a}::{jiange}]")
    for ori_image_path_, att_image_path_ in zip(ori_images_path, att_images_path):
        x = ori_image_path_.split('\\')[2][9]
        y = att_image_path_.split('\\')[2][9]
        if x != 'o' or y != 'a' or ori_image_path_.split('\\')[2][0:8] != att_image_path_.split('\\')[2][0:8]:
            raise IndexError('bupipei,huotupianbufu')
    att_images_path = att_images_path[a::jiange]
    ori_images_path = ori_images_path[a::jiange]
    images_num = len(ori_images_path)
    ii_top1 = 0
    ii_top2 = 0

    for ori_image_path, att_image_path in zip(ori_images_path, att_images_path):
        image_ori, image_att, ori_class, att_class, image_name = load_results_image(ori_image_path, att_image_path)
        image_ori = image_ori/255
        image_att = image_att /255
        # img, blurred_img, logitori = Get_blurred_img(image_att, att_class, model, resize_shape=(64, 64),
        #                                                      Gaussian_param=[5, 0.5],
        #                                                      Median_param=3, blur_type='Mixed')
        img, blurred_img = Get_blurred_img(image_att, att_class, model, resize_shape=resize_shape_,
                                                             Gaussian_param=[3, 50],
                                                             Median_param=11, blur_type='Gaussian')
        mask, upsampled_mask, imgratio, curvetop_att, curve1, curve2_attclass_cate, curve3_oriclass_cate,category, curvetop_ori,class_output,pred_top2 = Integrated_Mask(img, blurred_img, model,
                                                                                                         att_class,
                                                                                                          image_name,
                                                                                                         max_iterations=4,
                                                                                                         integ_iter=5,
                                                                                                         tv_beta=2,
                                                                                                         l1_coeff=0,
                                                                                                         tv_coeff=0,
                                                                                                         size_init=size_init_,
                                                                                                         ori_label=ori_class   )
        print(pred_top2,"ori_label:",ori_class,"att_label",att_class)
        pred_66_top2=50
        if pred_top2[0] == att_class:
            pred_66_top2=int(pred_top2[1])
        else:
            pred_66_top2=int(pred_top2[0])

        if pred_66_top2 == ori_class:
            ii_top2+=1
        if int(pred_top2[0]) == ori_class:
            ii_top1+=1

        print(curve2_attclass_cate)
        upsampled_mask = tf.squeeze(upsampled_mask).numpy()
        result_mask = (upsampled_mask-np.min(upsampled_mask))/(np.max(upsampled_mask)-np.min(upsampled_mask))
        image_saved_name = image_name[:-4] + '_matrix.csv'
        image_saved_path = os.path.join(resultimages_savedpath, image_saved_name)
        heatmap = cv2.applyColorMap(np.uint8(255 * result_mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

    acc_top1 = ii_top1/images_num
    acc_top2 = ii_top2/images_num
    print("acc_top1:",acc_top1,"acc_top2:",acc_top2)



