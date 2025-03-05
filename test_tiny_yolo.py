import os
import time
import glob
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners
from utils.yolo_utils import *
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Int8
import rclpy
from rclpy.node import Node

class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")

        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.keras.backend.get_session()

        self.yolo_model = load_model("/home/mohamedoutahar/Desktop/work/carla/object_Detect/src/tiny_b/model_data/tiny_yolo.h5")
        #yolo_model.summary()
        
        self.class_names = read_classes("/home/mohamedoutahar/Desktop/work/carla/object_Detect/src/tiny_b/model_data/yolo_coco_classes.txt")
        anchors = read_anchors("/home/mohamedoutahar/Desktop/work/carla/object_Detect/src/tiny_b/model_data/yolo_anchors.txt")
        # Generate colors for drawing bounding boxes.
        self.colors = generate_colors(self.class_names)

        '''
        # image detection
        image_file = "dog.jpg"
        image_path = "images/"
        image_shape = np.float32(cv2.imread(image_path + image_file).shape[:2])

        yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
        scores, boxes, classes = yolo_eval(yolo_outputs, image_shape=image_shape)

        # Start to image detect
        out_scores, out_boxes, out_classes = image_detection(sess, image_path, image_file, colors)
        '''
        
        # video detection
 
        #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 288) # 設計解析度
        #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
        #print('WIDTH', camera.get(3), 'HEIGHT', camera.get(4))
        #print('FPS', camera.get(5))


        #image_shape = np.float32(camera.get(4)), np.float32(camera.get(3))
        yolo_outputs = yolo_head(self.yolo_model.output, anchors, len(self.class_names))
        self.scores, self.boxes, self.classes = self.yolo_eval(yolo_outputs, score_threshold=.7)
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(Image, '/image', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, 'object_detection/result', 10)

 
        self.cl_pub = self.create_publisher(Int8, 'class', 10)


    def image_callback(self, data):

        self.rgb_image = self.bridge.imgmsg_to_cv2(data)
        #print(self.rgb_image.shape)
        image = self.video_detection(self.sess, self.rgb_image, self.colors)
        cv2.imshow('image', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return




    def yolo_eval(self, yolo_outputs, image_shape=(600., 800.), max_boxes=10, score_threshold=.6, iou_threshold=.5):    
        # Retrieve outputs of the YOLO model (≈1 line)
        box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

        # Convert boxes to be ready for filtering functions 
        boxes = yolo_boxes_to_corners(box_xy, box_wh)

        # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
        scores, boxes, classes = self.yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
        
        # Scale boxes back to original image shape.
        boxes = scale_boxes(boxes, image_shape) # boxes: [y1, x1, y2, x2]

        # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
        scores, boxes, classes = self.yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
        
        ### END CODE HERE ###
        
        return scores, boxes, classes

    def yolo_filter_boxes(self, box_confidence, boxes, box_class_probs, threshold = .6):    
        # Compute box scores
        box_scores = box_confidence * box_class_probs
        
        # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
        box_classes = K.argmax(box_scores, axis=-1)
        box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
        
        # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
        # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
        filtering_mask = box_class_scores >= threshold
        
        # Apply the mask to scores, boxes and classes
        scores = tf.boolean_mask(box_class_scores, filtering_mask)
        boxes = tf.boolean_mask(boxes, filtering_mask)
        classes = tf.boolean_mask(box_classes, filtering_mask)
        
        return scores, boxes, classes

    def yolo_non_max_suppression(self, scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
        max_boxes_tensor = K.variable(max_boxes, dtype='int32') # tensor to be used in tf.image.non_max_suppression()
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
        
        # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
        
        # Use K.gather() to select only nms_indices from scores, boxes and classes
        scores = K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)
        
        return scores, boxes, classes

    def image_detection(self, sess, image_path, image_file, colors):
        # Preprocess your image
        image, image_data = preprocess_image(image_path + image_file, model_image_size = (416, 416))
        
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={self.yolo_model.input:image_data, K.learning_phase():0})

        # Print predictions info
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        
        # Draw bounding boxes on the image file
        image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

        # Save the predicted bounding box on the image
        #image.save(os.path.join("out", image_file), quality=90)
        cv2.imwrite(os.path.join("out", "tiny_yolo_" + image_file), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return out_scores, out_boxes, out_classes

    def video_detection(self, sess, image, colors):
        resized_image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_scores, out_boxes, out_classes = sess.run([self.scores, self.boxes, self.classes], feed_dict={self.yolo_model.input:image_data, K.learning_phase():0})

        image = draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, self.colors)
        print(out_classes)

        return image
        
 


def main(args=None):
    rclpy.init(args=args)

    object_detection_node = ObjectDetection()
    rclpy.spin(object_detection_node)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
    object_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
