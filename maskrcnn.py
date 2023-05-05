import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def resize_masks(masks, boxes, image_shape):
    resized_masks = []
    for i in range(masks.shape[0]):
        box = boxes[i]
        mask = masks[i]
        ymin, xmin, ymax, xmax = box
        ymin = int(ymin * image_shape[0])
        ymax = int(ymax * image_shape[0])
        xmin = int(xmin * image_shape[1])
        xmax = int(xmax * image_shape[1])

        resized_mask = cv2.resize(mask, (xmax - xmin, ymax - ymin))
        full_size_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        full_size_mask[ymin:ymax, xmin:xmax] = resized_mask
        resized_masks.append(full_size_mask)
    return np.array(resized_masks)

# Download the pre-trained Mask R-CNN model from the TensorFlow Model Zoo
# URL: http://download.tensorflow.org/models/object_detection/tf2/YYYY/MM/DD/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz
# Replace 'PATH_TO_MODEL' with the path to the downloaded model
#use get_model.py
PATH_TO_MODEL = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/saved_model'

# Download the COCO labels file
# URL: https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt
# Replace 'PATH_TO_LABELS' with the path to the downloaded labels file
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

# Load the image
# Replace 'PATH_TO_IMAGE' with the path to your image
PATH_TO_IMAGE = 'clutter_1467.png'
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image_rgb = image
min_score_thresh=0.1
min_area = 1000  # Minimum area threshold, e.g., 1000 square pixels
max_area = 100000  # Maximum area threshold, e.g., 100000 square pixels

# Load the pre-trained model
model = tf.saved_model.load(PATH_TO_MODEL)

# Load the labels
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Run the model on the image
#input_tensor = tf.convert_to_tensor(image_rgb)
#input_tensor = input_tensor[tf.newaxis, ...]
input_tensor = tf.convert_to_tensor(np.expand_dims(image_rgb, axis=0), dtype=tf.uint8)
output_dict = model(input_tensor)

num_detections = int(output_dict.pop('num_detections'))
#output_dict = {key: value[0, :num_detections].numpy()
#               for key, value in output_dict.items()}

print('num_detections: ', num_detections)
'''             
for key, value in output_dict.items():
    if key != 'detection_masks':
        output_dict[key] = value[0, :num_detections].numpy()
    else:
        output_dict[key] = value.numpy()
'''
detection_boxes = output_dict['detection_boxes'][0, :num_detections].numpy()
#detection_classes = output_dict['detection_classes'][0, :num_detections].numpy().astype(np.int64)
detection_classes = output_dict['detection_classes'][0, :num_detections].numpy()
detection_classes = detection_classes.astype(np.int64)
detection_scores = output_dict['detection_scores'][0, :num_detections].numpy()
detection_masks = output_dict['detection_masks'][0].numpy()
detection_masks = (detection_masks * 255).astype(np.uint8)
detection_masks = resize_masks(detection_masks, detection_boxes, image.shape)

#output_dict['num_detections'] = num_detections
#output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

# Get the masks and scores
#detection_masks = output_dict['detection_masks']
#detection_scores = output_dict['detection_scores']

# Visualize the results
'''
viz_utils.visualize_boxes_and_labels_on_image_array(
    image,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    detection_scores,
    category_index,
    instance_masks=detection_masks,
    use_normalized_coordinates=True,
    line_thickness=8,
)
'''
image1 = image.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image1,
    detection_boxes,
    detection_classes,
    detection_scores,
    category_index,
    instance_masks=detection_masks,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=min_score_thresh,  # Set the minimum score threshold to display more objects
)
cv2.imwrite('image_with_objects_detected_and_segmented.png', image1)
output_directory = 'output_masks'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for i in range(detection_masks.shape[0]):
    mask = detection_masks[i]
    class_id = detection_classes[i]
    class_name = category_index[class_id]['name']
    score = detection_scores[i]
    
    if score > min_score_thresh:  # Only save masks for objects with a detection score higher than 0.5
        mask_filename = f"{output_directory}/{class_name}_{i}.png"
        cv2.imwrite(mask_filename, mask)


filtered_boxes = []
filtered_classes = []
filtered_scores = []
filtered_masks = []

for i in range(detection_boxes.shape[0]):
    box = detection_boxes[i]
    class_id = detection_classes[i]
    score = detection_scores[i]
    mask = detection_masks[i]

    if score > min_score_thresh:  # You can adjust the score threshold as needed
        # Calculate the bounding box dimensions
        ymin, xmin, ymax, xmax = box
        width = xmax - xmin
        height = ymax - ymin
        area = width * height * image.shape[0] * image.shape[1]

        # Check if the object's area is within the desired range
        if min_area <= area <= max_area:
            filtered_boxes.append(box)
            filtered_classes.append(class_id)
            filtered_scores.append(score)
            filtered_masks.append(mask)

# Convert the lists to numpy arrays
filtered_boxes = np.array(filtered_boxes)
filtered_classes = np.array(filtered_classes)
filtered_scores = np.array(filtered_scores)
filtered_masks = np.array(filtered_masks)

#cv2.imshow('Image with Objects Detected and Segmented', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.imwrite('image_with_objects_detected_and_segmented.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
image2 = image.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image2,
    filtered_boxes,
    filtered_classes,
    filtered_scores,
    category_index,
    instance_masks=filtered_masks,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=min_score_thresh,  # You can adjust the visualization score threshold as needed
)

cv2.imwrite('image_with_objects_detected_and_segmented_filtered.png', image2)

output_directory = 'output_masks_filtered'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for i in range(detection_masks.shape[0]):
    mask = detection_masks[i]
    class_id = detection_classes[i]
    class_name = category_index[class_id]['name']
    score = detection_scores[i]
    
    if score > min_score_thresh:  # Only save masks for objects with a detection score higher than 0.5
        mask_filename = f"{output_directory}/{class_name}_{i}.png"
        cv2.imwrite(mask_filename, mask)



# Assuming you have the following variables from your object detection script:
# image, filtered_boxes, filtered_classes, filtered_scores, filtered_masks, and category_index


# Set the mask color and transparency
mask_color = (0, 255, 0)  # Green
mask_color_hex = '#{:02x}{:02x}{:02x}'.format(*mask_color)  # Convert to hex color string
mask_alpha = 0.4

image3 = image.copy()
# Iterate through the detected objects
for i in range(filtered_boxes.shape[0]):
    box = filtered_boxes[i]
    class_id = filtered_classes[i]
    score = filtered_scores[i]
    mask = filtered_masks[i]

    # Resize the mask to match the size of the image
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Draw the mask on the image
    viz_utils.draw_mask_on_image_array(
        image3,
        resized_mask,
        color=mask_color_hex,  # Use the hex color string
        alpha=mask_alpha
    )

    # Draw the class label and score on the image
    ymin, xmin, ymax, xmax = box
    x, y = int(xmin * image.shape[1]), int(ymin * image.shape[0])
    label = f"{category_index[class_id]['name']}: {score:.2f}"
    image3 = cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 1)

# Show or save the image
cv2.imwrite('image_with_masks.png', image3)

