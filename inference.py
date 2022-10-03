from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
import cv2
import torch
import numpy as np
from torchvision import transforms

from models.common import DetectMultiBackend
from utils.general import non_max_suppression

import easyocr


class PlatesRecognitionInference(ABC):

    def __init__(self, inference_config, detection_transform=None):

        self.inference_config = inference_config
        self.detection_transform = detection_transform

        self.load_models()

    @abstractmethod
    def load_models(self):
        pass

    def load_image(self, img_path):

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def preprocces_image(self, img):
        if self.detection_transform is not None:
            img = self.detection_transform(img)
        img = transforms.ToTensor()(img)

        return img

    @abstractmethod
    def detect_plates(self, img):
        pass

    def recognized_plates(self, img):

        text = self.ocr_model.recognize(img, detail=0, allowlist="012334567890ABEKMHOPCTYX")[0]

        return text

    @torch.no_grad()
    def predict_image(self, img_path, visualize=False):

        img = self.load_image(img_path)
        torch_img = self.preprocces_image(img).to(self.inference_config['device'])

        bbox = self.detect_plates(torch_img)
        if bbox is not None:
            img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            text = self.recognized_plates(img_crop)
        else:
            text = ''

        if visualize:
            return img, bbox, text
        else:
            return text

    @abstractmethod
    def visualize_image(self, img_path):
        pass

    def predict_img_list(self, img_list):

        results = []
        for img in tqdm(img_list):
            results.append(self.predict_image(img))

        return results


class FRCNNRecognitionInference(PlatesRecognitionInference):

    def load_models(self):
        self.detection_model = torch.load(self.inference_config['detection_model'], map_location=self.inference_config['device']) \
           .to(self.inference_config['device']).eval()
        self.ocr_model = easyocr.Reader(**self.inference_config['ocr_model'])
   
    
    def preprocces_image(self, img):
        if self.detection_transform is not None:
            img = self.detection_transform(img)
        img = transforms.ToTensor()(img)
        return img

    def detect_plates(self, img):
        
        output = self.detection_model([img])[0]
        bboxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        if len(scores) > 0:
            bbox = bboxes[np.argmax(scores)].astype('int')
        else:
            bbox = None
        return bbox

    @torch.no_grad()
    def predict_image(self, img_path, visualize=False):
        
        img = self.load_image(img_path)
        torch_img = self.preprocces_image(img).to(self.inference_config['device'])
        bbox = self.detect_plates(torch_img)
        if bbox is not None:
            img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            try:
                text = self.recognized_plates(img_crop)
            except:
                text = '';
        else:
            text = ''
        if visualize:
            return img, bbox, text
        else:
            return text

    def visualize_image(self, img_path):

        img, bbox, text = self.predict_image(img_path, visualize=True)

        print('Recognized text:', text)
        fig, ax = plt.subplots(figsize=(20, 17))
        ax.imshow(img)
        rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                         linewidth=3, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    def predict_img_list(self, img_list):

        results = []
        for img in tqdm(img_list):
            results.append(self.predict_image(img))

        return results


class YoloRecognitionInference(PlatesRecognitionInference):

    def __init__(self, inference_config, detection_transform=None):

        super().__init__(inference_config, detection_transform)
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.classes = 1
        self.agnostic_nms = False
        self.max_det = 10
        self.img_shape = (640, 640)
        
    def load_models(self):
        self.detection_model = DetectMultiBackend(self.inference_config['detection_model'], device='cuda', dnn=False)
        self.detection_model.to(self.inference_config['device']).eval()
        self.ocr_model = easyocr.Reader(**self.inference_config['ocr_model'])


    def preprocces_image(self, img):
        self.img_shape = img.shape[:2]
        img = cv2.resize(img, (640, 640))
        if self.detection_transform is not None:
            img = self.detection_transform(img)
        img = transforms.ToTensor()(img)

        return img

    def detect_plates(self, img):

        output = self.detection_model(img.unsqueeze(0))
        output = non_max_suppression(output, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det= \
                                     self.max_det)
        boxes_unscaled = output[0][:, :4].cpu().numpy()
        dx = self.img_shape[1] / 640
        dy = self.img_shape[0] / 640
        bboxes = boxes_unscaled * [dx, dy, dx, dy]
        bboxes = bboxes.clip(min=0)

        return bboxes.astype('int')

    @torch.no_grad()
    def predict_image(self, img_path, visualize=False):

        img = self.load_image(img_path)
        torch_img = self.preprocces_image(img).to(self.inference_config['device'])
        texts = []
        bboxes = self.detect_plates(torch_img)
        if len(bboxes) != 0:
            for bbox in bboxes:
                img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                texts.append(self.recognized_plates(img_crop))
        else:
            texts = ''
        if visualize:
            return img, bboxes, texts
        else:
            return texts

    def visualize_image(self, img_path):

        img, bboxes, text = self.predict_image(img_path, visualize=True)

        print('Recognized text:', text)
        fig, ax = plt.subplots(figsize=(20, 17))
        ax.imshow(img)
        for bbox in bboxes:
            rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             linewidth=3, edgecolor='r', facecolor='none')

            ax.add_patch(rect)
        plt.plot()

    def predict_img_list(self, img_list):

        results = []
        for img in tqdm(img_list):
            results.append(self.predict_image(img))

        return results