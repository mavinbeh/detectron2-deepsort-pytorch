from distutils.util import strtobool
import os
from deep_sort import DeepSort
from detectron2_detection import Detectron2

class Detector(object):
    def __init__(self):
        self.detectron2 = Detectron2()
        deepsort_checkpoint = os.path.dirname(__file__)+"/deep_sort/deep/checkpoint/ckpt.t7"
        self.deepsort = DeepSort(deepsort_checkpoint, use_cuda=True)

    def detect_and_return(self, im):

        bbox_xcycwh, cls_conf, cls_ids, masks = self.detectron2.detect(im)

        if bbox_xcycwh is not None:
            # select class person
            mask = cls_ids == 0

            bbox_xcycwh = bbox_xcycwh[mask]
            bbox_xcycwh[:, 3:] *= 1.2

            cls_conf = cls_conf[mask]
            tracks = self.deepsort.update_and_return_tracks(bbox_xcycwh, cls_conf, im)

            results = {
                'rois': bbox_xcycwh, 
                'class_ids': cls_ids, 
                'scores': cls_conf, 
                'masks': masks
            }

            return results, tracks

if __name__ == "__main__":
    with Detector() as det:
        det.detect()
