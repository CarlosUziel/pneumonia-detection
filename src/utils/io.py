import tensorflow as tf


def load_tf_image(file_path: str) -> tf.Tensor:
    """Use tensorflow to load a .png image.

    Args:
        file_path: PNG image path to load.

    Returns:
        Image tensor.
    """
    png_img = tf.image.decode_png(tf.io.read_file(file_path), channels=3)
    return tf.image.convert_image_dtype(png_img, tf.float32)
