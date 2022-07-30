from grad_cam_intergradient import GradCAM
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from mytools import load_results_imagespath, load_results_image
import cv2
from collections import Counter

# def creat_deleted_pixel_image(model, image, which_class, input_image_shape, intensity, pixel_number_rate):
#
#
#     gradcam = GradCAM(model)
#      important_scope = gradcam.compute_heatmap(image, which_class, input_image_shape, pattern='select',
#                                                intensity=intensity)
#     # plt.imshow(np.array(important_scope*255, 'uint8'))
#     # plt.show()
#
#     important_scope = tf.cast(important_scope, tf.float32)
#     important_pixel_index = tf.where(important_scope > 0)
#     pixel_number = tf.cast(tf.round(pixel_number_rate * important_pixel_index.shape[0]), tf.int32)
#     if pixel_number == 0:
#         new_image = tf.constant(image)
#     # if important_pixel_index.shape[0] < pixel_number:
#   #      raise IndexError('pixel_number is larger than number of the important pixels:{}'.format(important_pixel_index.shape[0]))
#     else:
#         important_pixel_index = tf.random.shuffle(important_pixel_index)
#         select_pixel_index = important_pixel_index[0:pixel_number]
#         mask = tf.scatter_nd(select_pixel_index, tf.ones(pixel_number), shape=input_image_shape)
#         mask = tf.expand_dims(mask, 2)
#         mask = tf.tile(mask, (1, 1, 3))
#         a = tf.zeros(image.shape)
#         image_tensor = tf.constant(image)
#         new_image = tf.where(mask > 0, a, image_tensor)
#         # new_image = np.array(new_image, 'uint8')
#     return new_image


def creat_important_pixel_index(model, image, explain_label, input_image_shape, intensity, delete_pixel_number_rate, which_layer_f):
    gradcam = GradCAM(model, which_layer_f)
    important_scope ,_= gradcam.compute_heatmap(image, explain_label, input_image_shape,
                                              pattern='select', intensity=intensity)
    # plt.imshow(np.array(important_scope*255, 'uint8'))
    # plt.show()
    important_scope = tf.cast(important_scope, tf.float32)
    important_pixel_index = tf.where(important_scope > 0)
    delete_pixel_number = tf.cast(tf.round(delete_pixel_number_rate * important_pixel_index.shape[0]), tf.int32)
    return important_pixel_index, delete_pixel_number


def creat_deleted_pixel_image(important_pixel_index, image, input_image_shape, delete_pixel_number):
    if delete_pixel_number == 0:
        new_image = tf.constant(image)
    # if important_pixel_index.shape[0] < pixel_number:
    #     raise IndexError('pixel_number is larger than number of the important pixels:{}'.format(important_pixel_index.shape[0]))
    else:
        important_pixel_index = tf.random.shuffle(important_pixel_index)
        select_pixel_index = important_pixel_index[0:delete_pixel_number]
        mask = tf.scatter_nd(select_pixel_index, tf.ones(delete_pixel_number), shape=input_image_shape)
        if image.ndim == 3:
            mask = tf.expand_dims(mask, 2)
            mask = tf.tile(mask, (1, 1, 3))
        # a = tf.zeros(image.shape)
        # a = tf.constant(image)
        a = cv2.GaussianBlur(image, (3, 3), 50)
        image_tensor = tf.constant(image)
        new_image = tf.where(mask > 0, a, image_tensor)
        # new_image = np.array(new_image, 'uint8')
    return new_image


def calculate_acc(model, image, ori_label, explain_label, att_class_f, intensity, delete_pixel_number_rate,
                  batch_num=1, repeat_num=1, which_layer_f="conv2d_2"):

    input_image_shape = image.shape[0:2]
    important_pixel_index, delete_pixel_number = creat_important_pixel_index(model, image,
                                                                             explain_label, input_image_shape,
                                                                             intensity, delete_pixel_number_rate, which_layer_f)
    acc_list = []
    pred_list = []
    att_change_acc_list = []
    acc_top2_list = []
    deleted_pixel_images = []
    for i in range(batch_num):
        deleted_pixel_image = creat_deleted_pixel_image(important_pixel_index, image,
                                                        input_image_shape, delete_pixel_number)

        deleted_pixel_images.append(deleted_pixel_image/255)

        # print(tf.argmax(model.predict(tf.expand_dims(deleted_pixel_image/255, 0))))

        # deleted_pixel_image = np.array(deleted_pixel_image, 'uint8')
        # plt.imshow(deleted_pixel_image)
        # plt.show()
    deleted_pixel_images = tf.convert_to_tensor(deleted_pixel_images)
    if deleted_pixel_images.ndim == 3:
        deleted_pixel_images = tf.expand_dims(deleted_pixel_images, 3)
    prob = model.predict(deleted_pixel_images)
    pred = tf.argmax(prob, axis=1)
    pred_top2 = tf.math.top_k(prob, 2).indices
    pred_66_top2 = []
    pred_66_top1 = []

    for i in range(len(pred)):
        if pred[i] == att_class_f:
            pred_66_top2.append(int(pred_top2[i,1]))
        else:
            pred_66_top2.append(int(pred[i]))
    for i in range(len(pred)):
        pred_66_top1.append(int(pred[i]))


    correct_num_top2 = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(pred_top2[:, 0], ori_label), tf.logical_and(tf.equal(pred_top2[:, 0], att_class_f), tf.equal(pred_top2[:, 1], ori_label))), tf.int32))
    correct_num = tf.reduce_sum(tf.cast(tf.equal(pred, ori_label), tf.int32))
    change_num_att = tf.reduce_sum(tf.cast(tf.not_equal(pred, att_class_f), tf.int32))
    acc_list.append(correct_num/batch_num)
    acc_top2_list.append(correct_num_top2/batch_num)
    att_change_acc_list.append(change_num_att/batch_num)
    # print(pred_66)
    max_num_label_top2 = Counter(pred_66_top2).most_common(1)[0][0]
    max_num_label_top1 = Counter(pred_66_top1).most_common(1)[0][0]




    return max_num_label_top2, max_num_label_top1

def predict_for_several_images(root_f, model_path_f, which_layer_f,a_f, jiange_f,intensity = 0.3, delete_pixel_number_rate = 0.3):
    model_f = tf.keras.models.load_model(model_path_f)
    model_name_f = model_path_f.split('/')[-1][:-3]
    train_or_test = root_f.split('/')[-1]
    ori_images_path, att_images_path, att_method, dataset_name, episilon_f = load_results_imagespath(root_f)
    resultimages_savedpath = os.path.join('result_images/predict_for_delete_pixel_images_low_resolution/', dataset_name, model_name_f,
                                          which_layer_f, att_method + episilon_f, train_or_test)
    if not os.path.exists(resultimages_savedpath):
        os.makedirs(resultimages_savedpath)
    with open(resultimages_savedpath+"/already.txt", "a") as file:
        file.write(f"[{a_f}::{jiange_f}]")
    for ori_image_path_, att_image_path_ in zip(ori_images_path, att_images_path):
        x = ori_image_path_.split('\\')[2][9]
        y = att_image_path_.split('\\')[2][9]
        if x != 'o' or y != 'a' or ori_image_path_.split('\\')[2][0:8] != att_image_path_.split('\\')[2][0:8]:
            raise IndexError('bupipei,huotupianbufu')


    att_images_path = att_images_path[a_f::jiange_f]
    ori_images_path = ori_images_path[a_f::jiange_f]
    # print(len(att_images_path))
    current_num_top2 = 0
    current_num_top1 = 0
    for ori_image_path, att_image_path in zip(ori_images_path, att_images_path):
        image_ori, image_att, ori_class, att_class, image_name = load_results_image(ori_image_path, att_image_path)
        max_num_label_top2, max_num_label_top1 = calculate_acc(model_f, image_att, ori_class, att_class, att_class,intensity=intensity,delete_pixel_number_rate=delete_pixel_number_rate,batch_num=50, repeat_num=1,which_layer_f=which_layer_f)
        a_top2 = int(tf.equal(max_num_label_top2, ori_class))
        a_top1 = int(tf.equal(max_num_label_top1, ori_class))
        current_num_top2 += a_top2
        current_num_top1 += a_top1

    acc_top2 = current_num_top2 / len(att_images_path)
    acc_top1 = current_num_top1 / len(att_images_path)

    return acc_top2, acc_top1




model_path = 'model/resnet_20210104-011553acc81%epoch47.h5'
root = 'result_images/attack/miniimage/attack_bim_l2/episilon0.6/miniimage_ori_att_noi_image/test'
which_layer = "activation_15"


# model_path = 'model/cifar10_target_modelacc_0.81.h5'
# root = 'result_images/attack/cifar10/attack_fgsm_linf/episilon0.01/cifar10_ori_att_noi_image/test'
# which_layer = "activation_5"

# model_path = 'model/mnist_simple0.9945.h5'
# root = 'result_images/attack/mnist/attack_pgd_linf/episilon0.02/mnist_ori_att_noi_image/test'
# which_layer = "max_pooling2d_2"



"""
输入None即为最后一层4维层
for miniimage: "conv2d" "conv2d_1" "conv2d_2"......"conv2d_19" "activation_15"
for mnist: "conv2d" "conv2d_1" "conv2d_2" "max_pooling2d_2"
for cifar10: "conv2d  conv2d_1   conv2d_2   ...... conv2d_11  conv2d_12   activation_12 "
for cifar10: "conv2d  conv2d_1   conv2d_2   ...... activation_5  max_pooling2d_2   dropout_2 "
"""


acc_matic_top2 = np.zeros([11,11])
acc_matic_top1 = np.zeros([11,11])
for i in tqdm(range(11)):
    for j in range(11):
        intens = i/10
        delete_pixel_rate = j/10
        a = 24
        jiange = 50
        acc_top2, acc_top1 = predict_for_several_images(root, model_path, which_layer_f=which_layer, a_f=a, jiange_f=jiange,intensity=intens,delete_pixel_number_rate=delete_pixel_rate)
        acc_matic_top2[i, j] = acc_top2
        acc_matic_top1[i, j] = acc_top1

