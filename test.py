import mxnet as mx
import numpy as np
import net
import cv2


def preprocessing(input_img_path, input_height, input_width):

    input_image = cv2.imread(input_img_path)

    # Resize the image and convert to array of float32
    resized_image = cv2.resize(input_image, (input_height, input_width),
                               interpolation = cv2.INTER_CUBIC)
    image_data = np.array(resized_image, dtype='f')

    # Normalization [0,255] -> [0,1]
    image_data /= 255.

    # BGR -> RGB? The results do not change much
    # copied_image = image_data
    #image_data[:,:,2] = copied_image[:,:,0]
    #image_data[:,:,0] = copied_image[:,:,2]

    # Add the dimension relative to the batch size needed for the input placeholder "x"
    image_array = np.expand_dims(image_data, 0)  # Add batch dimension
    image_array = np.transpose(image_array, [0,3,1,2]) # NHWC -> NCHW

    print(image_array.shape)

    return image_array


def postprocessing(predictions, input_img_path, score_threshold, iou_threshold, input_height, input_width):

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    input_image = cv2.imread(input_img_path)
    input_image = cv2.resize(input_image, (input_height, input_width), interpolation = cv2.INTER_CUBIC)

    n_grid_cells = 13
    n_b_boxes = 5

    # Names and colors for each class
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127), 
                (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
                (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
                (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
                (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254), 
                (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
                (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
                (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254), 
                (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
                (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

    # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
    anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

    thresholded_predictions = []
    print('Thresholding on (Objectness score)*(Best class score) with threshold = {}'.format(score_threshold))

    # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
    # From now on the predictions are ORDERED and can be extracted in a simple way!
    # We have 13x13 grid cells, each cell has 5 B-Boxes, each B-Box have 25 channels with 4 coords, 1 Obj score , 20 Class scores
    # E.g. predictions[row, col, b, :4] will return the 4 coords of the "b" B-Box which is in the [row,col] grid cell
    predictions = np.transpose(predictions, [0, 2, 3, 1]) # NCHW -> NHWC
    predictions = np.reshape(predictions,(13,13,5,25))

    # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
    for row in range(n_grid_cells):
        for col in range(n_grid_cells):
            for b in range(n_b_boxes):

                tx, ty, tw, th, tc = predictions[row, col, b, :5]

                # IMPORTANT: (416 img size) / (13 grid cells) = 32!
                # YOLOv2 predicts parametrized coordinates that must be converted to full size
                # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
                center_x = (float(col) + sigmoid(tx)) * 32.0
                center_y = (float(row) + sigmoid(ty)) * 32.0

                roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
                roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

                final_confidence = sigmoid(tc)

                # Find best class
                class_predictions = predictions[row, col, b, 5:]
                class_predictions = softmax(class_predictions)

                class_predictions = tuple(class_predictions)
                best_class = class_predictions.index(max(class_predictions))
                best_class_score = class_predictions[best_class]

                # Compute the final coordinates on both axes
                left   = int(center_x - (roi_w/2.))
                right  = int(center_x + (roi_w/2.))
                top    = int(center_y - (roi_h/2.))
                bottom = int(center_y + (roi_h/2.))
      
                if( (final_confidence * best_class_score) > score_threshold):
                    thresholded_predictions.append([best_class,
                                                    final_confidence * best_class_score,
                                                    left, top, right, bottom])

    print('Printing {} B-boxes survived after score thresholding:'.format(len(thresholded_predictions)))
    for i, (best_class, _, l, t, r, b) in enumerate(thresholded_predictions):
        print('B-Box {} : {} {}'.format(i+1, [l, t, r, b], classes[best_class]))

    # Non maximal suppression
    print('Non maximal suppression with iou threshold = {}'.format(iou_threshold))
    nms_predictions = mx.contrib.nd.box_nms(mx.nd.array(thresholded_predictions), iou_threshold)
    nms_predictions = nms_predictions.asnumpy().astype('int')
    nms_predictions = nms_predictions[np.any(nms_predictions >= 0, axis=1)].tolist()

    # Print survived b-boxes
    print('Printing the {} B-Boxes survived after non maximal suppression:'.format(len(nms_predictions)))
    for i, (best_class, _, l, t, r, b) in enumerate(nms_predictions):
        print('B-Box {} : {} {}'.format(i+1, [l, t, r, b], classes[best_class]))

    # Draw final B-Boxes and label on input image
    for best_class, _, l, t, r, b in nms_predictions:

        color = colors[best_class]
        best_class_name = classes[best_class]

        # Put a class rectangle with B-Box coordinates and a class label on the image
        input_image = cv2.rectangle(input_image,(l, t),(r, b),color)
        cv2.putText(input_image, best_class_name, (int((l+r)/2), int((t+b)/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
  
    return input_image


def inference(sym, args, aux, preprocessed_image):
    # Forward pass of the preprocessed image into the network defined in the net.py file
    ctx = mx.cpu()
    preprocessed_image = mx.nd.array(preprocessed_image)

    args['data'] = preprocessed_image
    exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null')
    exe.forward(data=preprocessed_image)

    predictions = exe.outputs[0].asnumpy()
    print(predictions.shape)

    return predictions


# Definition of the paths
weights_path = './yolov2-tiny-voc.weights'
input_img_path = './dog.jpg'
output_image_path = './output.jpg'

# Definition of the parameters
input_height = 416
input_width = 416
score_threshold = 0.3
iou_threshold = 0.3

# Check for an existing checkpoint and load the weights (if it exists) or do it from binary file
print('Loading a network...')
sym = net.create_network()
args = net.load_weight(weights_path)
aux = {}

mx.visualization.print_summary(sym, shape={'data': (1,3,input_height,input_width)})

# Preprocess the input image
print('Preprocessing...')
preprocessed_image = preprocessing(input_img_path, input_height, input_width)

# Compute the predictions on the input image
print('Computing predictions...')
predictions = inference(sym, args, aux, preprocessed_image)

# Postprocess the predictions and save the output image
print('Postprocessing...')
output_image = postprocessing(predictions, input_img_path, score_threshold, iou_threshold,
                              input_height, input_width)

cv2.imwrite(output_image_path,output_image)
