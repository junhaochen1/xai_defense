import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from mytools import load_image_in_a_dir
import os




class GradCAM:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, classidx, upsample_size, eps=1e-5, pattern='', intensity=0.8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation

        image_blur = cv2.GaussianBlur(image, (3, 3), 50)
        # image_blur = tf.zeros(image.shape)
        images_list=[(1-i/10)*image+i/10*image_blur for i in range(11) if True]

        with tf.GradientTape() as tape:

            inputs = tf.cast(images_list, tf.float32)/255.


            (convouts, preds) = gradModel(inputs)  # preds before softmax

            loss = preds[:, classidx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convouts)
        # discard batch

        convouts = convouts[0]
        grads = tf.reduce_mean(grads,axis=0)

        norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convouts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) if np.max(cam) != 0.0 else cam / 1e-10
        cam = cv2.resize(cam, upsample_size, cv2.INTER_LINEAR)

        if pattern == 'select':
            cam[cam < intensity] = 0
            cam3_f = cam
        else:
            # convert to 3D

            cam3_f = np.expand_dims(cam, axis=2)
            cam3_f = np.tile(cam3_f, [1, 1, 3])

        return cam3_f, image_blur


def overlay_gradCAM(img, cam3_f):
    cam3_f = np.uint8(255 * cam3_f)
    cam3_f = cv2.applyColorMap(cam3_f, cv2.COLORMAP_JET)
    cam3_f = cv2.cvtColor(cam3_f, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype='float16')
    if img.ndim == 2:
        img = np.tile(np.expand_dims(img, 2).astype('uint8'), [1, 1, 3])
        a = 0.5
    else:
        a = 0.5
    new_img = 0.3 * cam3_f + a * img
    new_img = np.array(new_img * 255.0 / new_img.max(), 'uint8')
    return new_img





if __name__ == '__main__':
    # model = tf.keras.models.load_model('model/resnet_20210104-011553acc81%epoch47.h5')
    # image = 255*matplotlib.image.imread('result_images/predict_true/miniimage/train/6/00060003_predict_true.png')
    # Gradcam = GradCAM(model)
    # cam3 = Gradcam.compute_heatmap(image, 6, (64, 64))
    # new_image = overlay_gradCAM(image, cam3)
    # plt.imshow(new_image)
    # plt.show()

    dataset = "miniimage"
    which_layer = "activation_15"
    # ori_image_dir = "result_images/attack/miniimage/attack_bim_l2/episilon0.6/miniimage_only_attack_image/test"
    ori_image_dir = "D:/bishe/final/result_images/predict_true/miniimage/test"
    model_path = "model/resnet_20210104-011553acc81%epoch47.h5"
    model_name = model_path.split('/')[-1][:-3]
    t_or_t = ori_image_dir.split("/")[-1]
    model = tf.keras.models.load_model(model_path)
    Gradcam = GradCAM(model, which_layer)
    images, labels, identification_codes = load_image_in_a_dir(ori_image_dir)
    images = images[::50]
    labels = labels[::50]
    identification_codes = identification_codes[::50]

    for image,label,identification_code in zip(images, labels, identification_codes):
        image = image*255
        cam3 = Gradcam.compute_heatmap(image, label, (64, 64))
        new_image = overlay_gradCAM(image, cam3)
        new_image_saved_dir = os.path.join("result_images", "grad_cam_no_contrast", dataset, model_name, which_layer, t_or_t, str(label))
        if not os.path.exists(new_image_saved_dir):
            os.makedirs(new_image_saved_dir)
        new_image_name = identification_code+".png"
        plt.imshow(new_image)
        # plt.savefig(os.path.join(new_image_saved_dir, new_image_name))
        plt.show()
        # plt.close()


