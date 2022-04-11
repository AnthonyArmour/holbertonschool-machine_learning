from tensorflow.keras.utils import save_img
import tensorflow as tf

im1 = tf.keras.utils.load_img("images/before_random_crop.jpg")
im2 = tf.keras.utils.load_img("images/random_crop.jpg")

im1 = tf.image.resize(im1, im2.size)

save_img("./images/before_random_crop.jpg", im1)

