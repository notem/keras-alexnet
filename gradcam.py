import numpy as np
import tensorflow as tf
import keras
import cv2
import sys
import random
import os
from alexnet import load_dataset, default_model_name, default_model_dir


def load_model():
    """loads a trained model
    """
    if len(sys.argv) == 0:
        model_path = sys.argv[0]
    else:
        save_dir = os.path.join(os.getcwd(), default_model_dir)
        if not os.path.isdir(save_dir):
            raise EnvironmentError('Default model path does not exist!')
        model_name = default_model_name
        model_path = os.path.join(save_dir, model_name)
    return keras.models.load_model(model_path)


def deprocess_image(x):
    """
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def guided_backprop(model, image, target_layer):
    """modifies the model activation functions with a new gradient
    """
    def compile_saliency_function(model, target_layer):
        """
        :return:
        """
        model_input = model.input
        layer_output = model.get_layer(target_layer)
        max_output = keras.backend.max(layer_output, axis=3)
        saliency = keras.backend.gradients(keras.backend.sum(max_output), model_input)[0]
        return keras.backend.function([model_input, keras.backend.learning_phase()], [saliency])

    def modify_backprop(model, gradient_name):
        """recreates the model in which the guided back-prop gradient function overrides
        the usual relu activation functions
        :param model:
        :param gradient_name:
        :return:
        """
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': gradient_name}):

            # get layers that have an activation
            layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]

            # replace relu activation
            for layer in layer_dict:
                if layer.activation == keras.activations.relu:
                    layer.activation = tf.nn.relu

            # re-instantiate a new model with the tensorflow override
            new_model = load_model()
        return new_model

    # register the guided back-prop gradient function in tensorflow
    if "GuidedBackProp" not in tf.python.framework.ops._gradient_registry._registry:
        @tf.python.framework.ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype  # probably float
            # clever way to mask the gradient if either op.inputs or grad is negative
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

    # do guided backprop and create the saliency map
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model, target_layer)
    return saliency_fn([image, 0])[0]


def grad_cam(input_model, image, category_index, layer_name, height=224, width=224):
    """
    :param input_model:
    :param image:
    :param category_index:
    :param layer_name:
    :param width:
    :param height:
    :return:
    """
    y_c = input_model.output[0, category_index]             # prediction tensor for class
    conv_output = input_model.get_layer(layer_name).output  # tensor of the output of the last conv layer
    grads = keras.backend.gradients(y_c, conv_output)[0]    # output gradients tensor
    gradient_function = keras.backend.function([input_model.input], [conv_output, grads])  # computes output and gradient tensors

    output, grads_val = gradient_function([image])  # get gradient
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))  # compute weights as mean from gradient
    cam = np.dot(output, weights)   # compute activations from weights
    cam = np.maximum(cam, 0)        # do ReLU operation

    # c has been computed
    # process cam map into domain specific representation
    cam = cv2.resize(cam, (height, width))  # resize to image size
    cam = cam / np.max(cam)                 # create the heatmap
    return cam


def overlay_heatmap(image, heatmap):
    """
    :param image:
    :param heatmap:
    :return:
    """
    # Return to BGR [0..255] from the preprocessed image
    #image = image[0, :]
    #image -= np.min(image)
    #image = np.minimum(image, 255)

    # colorize the heatmap and merge into the image
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)


def main():
    # load model
    model = load_model()

    # load target image
    (x_train, y_train), (x_test, y_test) = load_dataset()
    index = random.randint(0, len(x_test))
    img = x_test[index]
    cv2.imwrite(index+".jpg", img)

    # make a prediction
    predictions = model.predict(img)
    predicted_class = keras.backend.argmax(predictions, axis=1)
    print("Supplied image was classified as [%u] by the model." % predicted_class)

    # apply grad-cam
    heatmap = grad_cam(model, img, predicted_class, "conv2d_5")
    cv2.imwrite(index+"_gradcam.jpg", overlay_heatmap(img, heatmap))

    # produce saliency map using guided backprop
    saliency = guided_backprop(model, img, "conv2d_5")
    cv2.imwrite(index+"_saliency.jpg", deprocess_image(saliency))

    # combine saliency map with heatmap
    guided_gradcam = saliency * heatmap[..., np.newaxis]
    cv2.imwrite(index+"_guided-gradcam.jpg", deprocess_image(guided_gradcam))


if __name__ == "__main__":
    # execute only if run as a script
    main()
