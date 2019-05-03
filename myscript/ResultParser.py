##################################################################
# Evaluate the result with COCO style of detection               #
#                                                                #
# @copyright    Copyright 2012-1019 Bytedance.Inc                #
# @license      Apache                                           #
# @author       Jiguo Li (jiguo.li@vipl.ict.ac.cn)               #
# @version      1.0                                              #
##################################################################

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt 
import skimage.io as io
import argparse
import numpy as np

class ResultParser(object):
    annType = ['segm','bbox','keypoints']
    def __init__(self, gt_ann_file, dt_result_file):
        self.cocoGt = COCO(gt_ann_file)
        self.cocoDt = self.cocoGt.loadRes(dt_result_file)
        self.imgIds = sorted(self.cocoGt.getImgIds())
        self.catIds = sorted(self.cocoGt.getCatIds())
        self.cats = self.cocoGt.cats
        self.coco_eval = COCOeval(self.cocoGt, self.cocoDt) #default iouType is 'segm'

    def visualize_result(self, img_id=None, show_gt=True):
        if img_id is None:
            img_id = self.imgIds[0]
        img = self.cocoGt.loadImgs(img_id)[0]
        I = io.imread(img['coco_url'])
        if show_gt:
            plt.subplot(2,1,1)
            plt.imshow(I)
            self._visualize_result(img, self.cocoGt)
            plt.subplot(2,1,2)
            plt.imshow(I)
            self._visualize_result(img, self.cocoDt)
            plt.show()
        else:
            plt.imshow(I)
            self._visualize_result(img, self.cocoDt)
            plt.show()


    @staticmethod
    def _visualize_result(img, coco:COCO):
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)

    def eval(self, annType='bbox', catIds=None, imgIds=None):
        coco_eval = self.coco_eval
        coco_eval.params.iouType = annType
        if catIds is not None:
            coco_eval.params.catIds = catIds
        if imgIds is not None:
            coco_eval.params.imgIds = imgIds
        coco_eval.evaluate()
        coco_eval.accumulate()
        iouThrList = [None, .5, .75, 'small', 'medium', 'large']
        ap_list = []
        for iouThr in iouThrList:
            ap_list.append(
                self._summarize(iouThr=iouThr, maxDets=self.coco_eval.params.maxDets[2])
            )
        return ap_list

    def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = self.coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        return mean_s

def get_parser():
    parser = argparse.ArgumentParser("ResultParser")
    parser.add_argument("--gt_ann_file", type=str, 
        default="/media/ubuntu/Elements/dataset/COCO/annotations/instances_val2017.json", 
        help="")
    parser.add_argument("--dt_ann_file", type=str,
        default="./results.pkl.json",
        help="")
    args, _ = parser.parse_known_args()
    return args


def main(args):
    parser = ResultParser(gt_ann_file=args.gt_ann_file, dt_result_file=args.dt_ann_file)
    cat_ap_list_all = []
    for catId in parser.catIds:
        ap_list = parser.eval(catIds=[catId])
        cat_ap_list_all.append(ap_list)
    
    results = [(catId, ap_list) for catId, ap_list in zip(parser.catIds, cat_ap_list_all)]
    results = sorted(results, key=lambda x: x[1][0])
    print("{:>15s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}".format('cat_name', '0.5:0.95', '0.5', '0.75', 'small', 'medium', 'large'))
    for catId, ap_list in results:
        print("{:>15s}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}".format(
            parser.cats[catId]['name'], ap_list[0], ap_list[1], ap_list[2], ap_list[3], ap_list[4], ap_list[5])
            )
        

if __name__=="__main__":
    args = get_parser()
    main(args)