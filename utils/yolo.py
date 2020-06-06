import cv2
import os


def initialise_yolo():
    '''
    Function to initialise the weights, architecture, and labels for the yolo model

    :return: yolo model, yolo model output layer names, model labels (ball, person, etc.)
    '''
    yolo_dir = os.path.abspath("./yolo")

    # load the labels, weights, and config for the yolo model
    labels_path = os.path.sep.join([yolo_dir, "coco.names"])
    weights_path = os.path.sep.join([yolo_dir, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_dir, "yolov3.cfg"])

    # load the labels (as list), and model
    labels = open(labels_path).read().strip().split("\n")
    yolo_model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # get the output layers
    layer_names = yolo_model.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]

    return yolo_model, layer_names, labels
