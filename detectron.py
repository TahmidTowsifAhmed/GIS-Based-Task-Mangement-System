from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import os

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.model_type = model_type
        
        if model_type=="LVIS": #LVIS segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)
    
    def detect_and_save(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
                file_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)
                
                if os.path.isfile(file_path):
                    self.onImage(file_path, output_path)
                else:
                    print("Invalid file path:", file_path)
    
    def onImage(self, image_path, output_path):
        image = cv2.imread(image_path)
        
        if image is None:
            print("Unable to read image:", image_path)
            return
        
        predictions = self.predictor(image)
        instances = predictions["instances"].to("cpu")
        
        if len(instances) == 0:
            print("No object detected in image:", image_path)
            return
        
        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TEST[0]),
                         instance_mode=ColorMode.SEGMENTATION)
        output = viz.draw_instance_predictions(instances).get_image()[:, :, ::-1]
        
        cv2.imwrite(output_path, output)
        print("Output saved to:", output_path)


# Set input and output directories
input_dir = "hilleroed"
output_dir = "hilleroed_output"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize detector object
detector = Detector(model_type="LVIS")

# Run detection on images from input directory and save the detected images in the output directory
detector.detect_and_save(input_dir, output_dir)


