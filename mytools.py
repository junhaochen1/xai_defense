import  os, glob
import  random, csv
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.image
import matplotlib.pyplot as plt
import shutil
import scipy.io


def load_csv(root, filename, name2label):
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):  # 如果没有csv文件，建立空images列表
        images = []
        for name in name2label.keys():                    # 遍历name2label字典中的所有键 即 5种类名
            # 'miniimage\\n01532829\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))   # root:miniimage name1:fff   os.path.join(root, name, '*.png') 即 'pokemon\\fff\\*.png'
            images += glob.glob(os.path.join(root, name, '*.jpg'))  # glob.glob 返回miniimage\\fff\\ 文件夹中的所有.png格式的文件的路径
            images += glob.glob(os.path.join(root, name, '*.jpeg'))

        # 600, 'miniimage\\n01532829\\00000000.png'
        print(len(images), images)

        random.shuffle(images)      # 将这个列表彻底打乱
        with open(os.path.join(root, filename), mode='w', newline='') as f: # 第一个参数文件的地址和名字 这里为miniimage\xxx.csv   newline=''如果不加则会行与行之间出现空行
            writer = csv.writer(f)                                     #  mode：w	打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
            for img in images:  # 'miniimage\\n01532829\\00000000.png'     img指的还是图片的路径
                name = img.split(os.sep)[-2]     # 即在\\处将路径拆成列表 取倒数第二个 即类名 n01532829
                label = name2label[name]     # 在这个字典中 用name即键名取数字表示的类别，结果如下
                # 'miniimage\\n01532829\\00000000.png', 0
                writer.writerow([img, label])   # 应该是一行一行写的意思  也可直接写csv.writer(f).writerow([img, label])
            print('written into csv file:', filename)

    # read from csv file                      读取csv文件
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:                              #一行行读
            # 'miniimage\\n01532829\\00000000.png', 0
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)      # 断言，作用类似if

    return images, labels

def load_images(root, mode='train'):
    # 创建数字编码表
    name2label = {}  # "sq...":0
    for name in sorted(os.listdir(os.path.join(root))):  # name遍历root路径中的所有文件和文件夹，
        if not os.path.isdir(os.path.join(root, name)):  # 如果root/name 这个路径不是文件夹
            continue                                      # 那么跳过下一步，
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())   # name为键，后面的为键的赋值， name1：0   name2:1 以此类推

    # 读取Label信息
    # [file1,file2,], [3,1]
    images, labels = load_csv(root, 'images.csv', name2label)        # 读出的images为列表，里面存的是图片的地址  labels存的是0~3

    if mode == 'train':  # 60%                 如果是训练模式则取前60%的列表
        images = images[:int(0.8 * len(images))]
        labels = labels[:int(0.8 * len(labels))]
    elif mode == 'all':  # 20% = 60%->80%
        pass
    else:  # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label

img_mean = tf.constant([0, 0, 0])
img_std = tf.constant([0, 0, 0])
def normalize(x, mean=img_mean, std=img_std):     # 对图片进行正则化
    # x: [224, 224, 3]          根据broadcast规则  会从后面开始对齐  将mean从[3]转为[224,224,3]
    # mean: [224, 224, 3], std: [3]
    x = (x - mean)/std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    x = x * std + mean
    return x

def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)  #  读取文件
    x = tf.image.decode_jpeg(x, channels=3) # RGBA   #  将一张图象还原成一个三维矩阵uint8。需要解码的过程。
    x = tf.image.resize(x, [244, 244])

    # data augmentation, 0~255
    # x = tf.image.random_flip_up_down(x)
    x= tf.image.random_flip_left_right(x)   # 输出image沿着第二维翻转的内容
    x = tf.image.random_crop(x, [224, 224, 3])  # 将一个形状 size 部分从 value 中切出

    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x)

    y = tf.convert_to_tensor(y)

    return x, y

def random_aspect(image):
    height, width, _ = np.shape(image)
    aspect_ratio = np.random.uniform(0.8, 1.25)
    if height < width:
        resize_shape = (int(width * aspect_ratio), height)
    else:
        resize_shape = (width, int(height * aspect_ratio))
    return cv2.resize(image, resize_shape)

def random_size(image):
    height, width, _ = np.shape(image)
    target_size = np.random.randint(73, 85)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width
    resize_shape = (int(width * size_ratio), int(height * size_ratio))  # width and height in cv2 are opposite to np.shape()
    return cv2.resize(image, resize_shape)

def random_crop(image, net_input_shape=(64, 64, 3)):
    height, width, _ = np.shape(image)

    input_height, input_width, _ = net_input_shape
    crop_x = np.random.randint(0, width - input_width)
    crop_y = np.random.randint(0, height - input_height)
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

def random_hsv(image):
    random_h = np.random.uniform(-36, 36)
    random_s = np.random.uniform(0.8, 1.2)
    random_v = np.random.uniform(0.8, 1.2)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = (image_hsv[:, :, 0] + random_h) % 180.0  # hue
    image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * random_s, 255)  # saturation
    image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * random_v, 255)  # brightness
    image_hsv = image_hsv.astype('uint8')
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

def random_gausian(image):
    noise = np.random.normal(0, 1, (64, 64, 3))
    # image = image.astype('float32')
    image = image + noise
    image = np.maximum(np.minimum(image, 255), 0)
    # image = image.tolist('int8')
    # image = np.array(image)
    image = image.astype('uint8')
    return image


def convert_synthetic_digits_dataset_to_mat_format(root: str = "datasets/synthetic_digits",
                                                   saved_path: str = "datasets",
                                                   resize: tuple = (64, 64),
                                                   mat_dataset_name_f="synthetic_digits_trainingset.mat") -> None:
    training_dir = root
    class_name_f = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    trainingset = tf.keras.preprocessing.image_dataset_from_directory(training_dir,
                                                                      class_names=class_name_f,
                                               labels="inferred",
                                               label_mode="int",
                                               batch_size=64,
                                               shuffle=True,
                                               image_size=resize)

    train_x = np.concatenate([data[0].numpy() for data in trainingset])
    train_y = np.concatenate([data[1].numpy() for data in trainingset])
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    scipy.io.savemat(os.path.join(saved_path, mat_dataset_name_f), dict(X=train_x, y=train_y))


def save_image(image, image_saved_dirpath, image_file_name ):
    '''
    输入的图片为0到1的浮点数的tensor,图片要存的文件夹路径，图片名字
    :return:
    '''
    image = image*255
    image = image[:, :, ::-1]
    image_np = np.array(image, dtype='uint8')
    image_file_path = os.path.join(image_saved_dirpath, image_file_name)
    if not os.path.exists(image_saved_dirpath):
        os.makedirs(image_saved_dirpath)
    cv2.imwrite(image_file_path, image_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def load_results_imagespath(root):
    '''

    :param root:
    :return: 返回本文件中所有原图片的列表,所有攻击图片的列表，攻击方式，数据集名称
    '''
    #result_images/attack/miniimage/attack_bim_l2/episilon0.6/miniimage_ori_att_noi_image/test
    att_method = root.split('/')[3][7:]
    episilon_f = root.split('/')[4]
    dataset_name = root.split('/')[2]
    label_list = os.listdir(root)
    images_path = []
    for label in label_list:
        images_path += glob.glob(os.path.join(root, label, '*.png'))
    if root.split('/')[5][-22:] == 'adv_ori_true_att_false':
        att_images_path = images_path[0::2]
        ori_images_path = images_path[1::2]
    else:
        att_images_path = images_path[0::3]
        ori_images_path = images_path[2::3]

    return ori_images_path, att_images_path, att_method, dataset_name, episilon_f
# 文件夹中为原始攻击噪音图片


def load_results_image(ori_image_path, att_image_path):
    '''

    :param ori_image_path:
    :param att_image_path:
    :return: 返回路径地址的原始图片，攻击图片，原始分类，攻击分类，去除(攻击还是原始)标签的图像名字
    '''
    image_ori = 255 * matplotlib.image.imread(ori_image_path)
    image_att = 255 * matplotlib.image.imread(att_image_path)
    # matplotlib读取png的图片时读入的是除了255的  可能是因为用cv2存的png

    ori_class = int(os.path.basename(att_image_path)[21:23])
    att_class = int(os.path.basename(att_image_path)[31:33])
    image_name = ori_image_path.split('\\')[-1][0:8] + ori_image_path.split('\\')[-1][13:]
    return image_ori, image_att, ori_class, att_class, image_name

def load_results_image_for_jpg(ori_image_path, att_image_path):
    '''

    :param ori_image_path:
    :param att_image_path:
    :return: 返回路径地址的原始图片，攻击图片，原始分类，攻击分类，去除(攻击还是原始)标签的图像名字
    '''
    image_ori =  matplotlib.image.imread(ori_image_path)
    image_att =  matplotlib.image.imread(att_image_path)
    # matplotlib读取png的图片时读入的是除了255的  可能是因为用cv2存的png

    ori_class = int(os.path.basename(att_image_path)[21:23])
    att_class = int(os.path.basename(att_image_path)[31:33])
    image_name = ori_image_path.split('\\')[-1][0:8] + ori_image_path.split('\\')[-1][13:]
    return image_ori, image_att, ori_class, att_class, image_name

def make_csv_for_image_from_different_directory(root_ori_train_f, root_ori_test_f, root_att_train_f, root_att_test_f):
    # root_ori_train = 'datasets/miniimage_aug_train'
    # root_ori_test = 'datasets/miniimage_aug_test'
    # root_att_train = 'result_images/attack/miniimage/attack_bim_l2/episilon0.6/miniimage_only_attack_image/train'
    # root_att_test = 'result_images/attack/miniimage/attack_bim_l2/episilon0.6/miniimage_only_attack_image/test'

    datasets_name_f = root_att_train_f.split('/')[2]
    dir_name_part = '{}_adv_{}_e{}'.format(datasets_name_f, root_att_train_f.split('/')[3][7:], root_att_train_f.split('/')[4][8:])
    csv_saved_root = 'datasets/{}_att'.format(dir_name_part)
    csv_file_name_adv_train = '{}_train.csv'.format(dir_name_part)
    csv_file_name_adv_test = '{}_test.csv'.format(dir_name_part)
    csv_file_name_ori_test = '{}_ori_test.csv'.format(datasets_name_f)

    if not os.path.exists(csv_saved_root):
        os.makedirs(csv_saved_root)

    ori_train_list = load_imagepaths_in_a_dir(root_ori_train_f)
    ori_test_list = load_imagepaths_in_a_dir(root_ori_test_f)
    att_train_list = load_imagepaths_in_a_dir(root_att_train_f)
    att_test_list = load_imagepaths_in_a_dir(root_att_test_f)

    adv_train_list = ori_train_list + att_train_list
    random.shuffle(adv_train_list)

    with open(os.path.join(csv_saved_root, csv_file_name_adv_train), mode='w', newline='') as f:
        writer = csv.writer(f)
        for image_path in adv_train_list:
            if image_path.split('\\')[-1][6:9] == 'ori':
                label = int(image_path.split('\\')[-1][18:20])
            else:
                label = int(image_path.split('\\')[-1][2:4])
            writer.writerow([image_path, label])
    with open(os.path.join(csv_saved_root, csv_file_name_adv_test), mode='w', newline='') as f:
        writer = csv.writer(f)
        for image_path in att_test_list:
            label = int(image_path.split('\\')[-1][2:4])
            writer.writerow([image_path, label])
    with open(os.path.join(csv_saved_root, csv_file_name_ori_test), mode='w', newline='') as f:
        writer = csv.writer(f)
        for image_path in ori_test_list:
            label = int(image_path.split('\\')[-1][18:20])
            writer.writerow([image_path, label])



def from_csv_load_imagepath_label(csvfile_path):
    image_pathes = []
    image_labels = []
    with open(csvfile_path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            image_path, label = row
            label = int(label)
            image_pathes.append(image_path)
            image_labels.append(label)
    return image_pathes, image_labels



def load_image_in_a_dir(root_f):
    dirs_f = os.listdir(root_f)
    image_pathes_f = []
    images_f = []
    labels_f = []
    identification_codes_f = []
    for dir_f in dirs_f:
        image_pathes_f += glob.glob(os.path.join(root_f, dir_f, '*.png'))
    for image_path_f in image_pathes_f:
        images_f.append(plt.imread(image_path_f))
        labels_f.append(int(os.path.basename(image_path_f)[2:4]))
        identification_codes_f.append(os.path.basename(image_path_f)[0:8])
    return images_f, labels_f, identification_codes_f
#  返回图片，标签，图片识别码列表  文件夹中为统一图片


def load_imagepaths_in_a_dir(root_f):
    dirs_f = os.listdir(root_f)
    image_pathes_f = []
    for dir_f in dirs_f:
        image_pathes_f += glob.glob(os.path.join(root_f, dir_f, '*.png'))
    return image_pathes_f
#  返回图片路径 文件夹中为统一图片


def preprocess2(x, y, a, b, c, d, e):
    x = tf.cast(x, tf.float32)/255
    y = tf.cast(y, tf.float32)/255
    a = tf.cast(a, tf.int32)
    b = tf.cast(b, tf.int32)
    return x, y, a, b, c, d, e
def select_images_ori_true_att_false(root_f,model_path_f):
    ori_images_path_f, att_images_path_f, att_method_f, dataset_name_f, episilon_f = load_results_imagespath(root_f)
    model_f = tf.keras.models.load_model(model_path_f)
    images_ori_list_f = []
    images_att_list_f = []
    ori_class_list_f = []
    old_att_class_list_f = []
    image_name_list_f = []

    for ori_image_path_f, att_image_path_f in zip(ori_images_path_f, att_images_path_f):
        image_ori_f, image_att_f, ori_class_f, old_att_class_f, image_name_f = load_results_image(ori_image_path_f, att_image_path_f)
        images_ori_list_f.append(image_ori_f)
        images_att_list_f.append(image_att_f)
        ori_class_list_f.append(ori_class_f)
        old_att_class_list_f.append(old_att_class_f)
        image_name_list_f.append(image_name_f)
    db_f = tf.data.Dataset.from_tensor_slices((images_ori_list_f, images_att_list_f, ori_class_list_f, old_att_class_list_f, image_name_list_f, ori_images_path_f, att_images_path_f))
    db_f = db_f.batch(32).map(preprocess2)
    # result_new_att_class_list_f = []
    # result_old_att_class_list_f = []
    # result_image_name_list_f = []
    for images_ori_list_ff, images_att_list_ff, ori_class_list_ff, att_class_list_ff, image_name_list_ff, ori_images_path_ff, att_images_path_ff in db_f:
        predict_ori_true_list_ff = tf.cast(tf.argmax(model_f.predict(images_ori_list_ff), axis=1), tf.int32) == ori_class_list_ff
        predict_att_new_label_list_ff = tf.cast(tf.argmax(model_f.predict(images_att_list_ff), axis=1), tf.int32)
        predict_att_false_list_ff = predict_att_new_label_list_ff != ori_class_list_ff
        predict_ori_true_att_false_list_ff = tf.logical_and(predict_ori_true_list_ff, predict_att_false_list_ff)
        if not any(predict_ori_true_att_false_list_ff):
            continue
        # result_new_att_class_list_f.append(predict_att_new_label_list_ff[predict_ori_true_att_false_list_ff])
        # result_old_att_class_list_f.append(att_class_list_ff[predict_ori_true_att_false_list_ff])
        # result_image_name_list_f.append(image_name_list_ff[predict_ori_true_att_false_list_ff])
        # result_new_att_class_list_f = np.array(predict_att_new_label_list_ff[predict_ori_true_att_false_list_ff], 'str')
        # result_old_att_class_list_f = np.array((att_class_list_ff[predict_ori_true_att_false_list_ff], 'str'))
        # result_image_name_list_f = np.array((image_name_list_ff[predict_ori_true_att_false_list_ff], 'str'))
        # result_ori_images_path_ff = np.array((ori_images_path_ff[predict_ori_true_att_false_list_ff], 'str'))
        # result_att_images_path_ff = np.array((att_images_path_ff[predict_ori_true_att_false_list_ff], 'str'))
        result_new_att_class_list_f = predict_att_new_label_list_ff[predict_ori_true_att_false_list_ff]
        result_old_att_class_list_f = att_class_list_ff[predict_ori_true_att_false_list_ff]
        result_image_name_list_f = image_name_list_ff[predict_ori_true_att_false_list_ff]
        result_ori_images_path_ff = ori_images_path_ff[predict_ori_true_att_false_list_ff]
        result_att_images_path_ff = att_images_path_ff[predict_ori_true_att_false_list_ff]
        for i in range(len(result_new_att_class_list_f)):
            result_new_att_class = int(result_new_att_class_list_f[i])
            result_old_att_class = int(result_old_att_class_list_f[i])
            result_image_name = str(np.array(result_image_name_list_f[i], 'str'))
            result_ori_image_path_f = str(np.array(result_ori_images_path_ff[i], 'str'))
            result_att_image_path_f = str(np.array(result_att_images_path_ff[i], 'str'))
            new_ori_image_name = result_image_name[0:8]+'_ori_'+result_image_name[8:-6]+'%02doldattclass%02d.png' % (result_new_att_class, result_old_att_class)
            new_att_image_name = result_image_name[0:8] + '_att_' + result_image_name[8:-6] + '%02doldattclass%02d.png' % (result_new_att_class, result_old_att_class)
            new_ori_image_dir_path = os.path.join(root_f.replace('ori_att_noi_image', 'ori_att_noi_image_adv_ori_true_att_false'), '%02d' % (result_new_att_class))
            if not os.path.exists(new_ori_image_dir_path):
                os.makedirs(new_ori_image_dir_path)
            new_ori_image_path = os.path.join(new_ori_image_dir_path, new_ori_image_name)
            new_att_image_path = os.path.join(new_ori_image_dir_path, new_att_image_name)

            shutil.copyfile(result_ori_image_path_f, new_ori_image_path)
            shutil.copyfile(result_att_image_path_f, new_att_image_path)
# 从 攻击图片的test中挑选原图在对抗训练模型预测正确，攻击图片在模型上预测错误的组合照片


def mat2png(root_f, dataset_name_f, train_or_test_f):
    dataset_f = scipy.io.loadmat(root_f)
    if train_or_test_f == 'train':
        j = 0
    else:
        j = 80000
    images_f, labels_f = dataset_f["X"], np.squeeze(dataset_f["y"])
    for image_f, label_f in zip(images_f, labels_f):
        image_saved_dirpath = 'datasets/{}_{}/{}'.format(dataset_name_f, train_or_test_f, label_f)
        if not os.path.exists(image_saved_dirpath):
            os.makedirs(image_saved_dirpath)
        image_name = '%05d_ori_oriclass%02d.png' % (j, label_f)
        image = image_f[:, :, ::-1]
        image_np = np.array(image, dtype='uint8')
        image_file_path = os.path.join(image_saved_dirpath, image_name)
        cv2.imwrite(image_file_path, image_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        j = j+1
# 从mat中提取出png图片

def topmaxPixel(HattMap, thre_num):
    b = np.argsort(HattMap.ravel())
    c = b[:thre_num ]
    ii = np.unravel_index(c, HattMap.shape)
    OutHattMap = HattMap*0
    OutHattMap[ii] = 1
    img_ratio = np.sum(OutHattMap) / OutHattMap.size
    OutHattMap = 1 - OutHattMap
    return OutHattMap, img_ratio

def tv_norm(input, tv_beta):
    img = input[0,:, :, 0]
    row_grad = tf.reduce_mean(tf.abs(img[:-1, :] - img[1:, :])**(tv_beta))
    col_grad = tf.reduce_mean(tf.abs(img[:, :-1] - img[:, 1:])**(tv_beta))
    return row_grad + col_grad


if __name__ == '__main__':

    # 从 攻击图片的test或train中挑选原图在对抗训练模型预测正确，攻击图片在模型上预测错误的组合照片
    # root = 'result_images/attack/cifar10/attack_bim_l2/episilon0.6/cifar10_ori_att_noi_image/test'
    # model_path = 'model/cifar10_adv_bim_l2_e0.6.h5'
    # # model_path ='model/resnet_20210104-011553acc81%epoch47.h5'
    # select_images_ori_true_att_false(root, model_path)

    # # 从mat中提取出png图片
    # root = 'datasets/mnist_testset.mat'
    # mat2png(root, 'mnist', train_or_test_f='test')


    #从不同的文件夹制作csv列表用于强化训练
    # root_ori_train = 'datasets/miniimage_aug_train'
    # root_ori_test = 'datasets/miniimage_aug_test'
    # root_att_train = 'result_images/attack/miniimage/attack_bim_l2/episilon0.6/miniimage_only_attack_image/train'
    # root_att_test = 'result_images/attack/miniimage/attack_bim_l2/episilon0.6/miniimage_only_attack_image/test'
    root_ori_train = 'datasets/mnist_train'
    root_ori_test = 'datasets/mnist_test'
    root_att_train = 'result_images/attack/mnist/attack_bim_l2/episilon1.2/mnist_only_attack_image/train'
    root_att_test = 'result_images/attack/mnist/attack_bim_l2/episilon1.2/mnist_only_attack_image/test'
    make_csv_for_image_from_different_directory(root_ori_train, root_ori_test, root_att_train, root_att_test)


    # 将文件夹的图片存为mat格式
    # root = 'datasets/miniimage_test'
    # resize = (64, 64)
    # convert_synthetic_digits_dataset_to_mat_format(root, "datasets", resize, mat_dataset_name_f="miniimage_testset.mat")



    pass