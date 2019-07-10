import os
import sys
from pathlib import Path
import cv2
import torch
from config import config
import numpy as np
import tqdm

openpose_dir = Path('src/pytorch_Realtime_Multi_Person_Pose_Estimation')
sys.path.append(str(openpose_dir))
sys.path.append("src/utils")

# FIXME: src~を消す。
from src.pytorch_Realtime_Multi_Person_Pose_Estimation.network.rtpose_vgg import get_model
from src.pytorch_Realtime_Multi_Person_Pose_Estimation.evaluate.coco_eval import get_multiplier, get_outputs

class Condition(object):

    def __init__(self):
        self.model = get_model('vgg19')
        weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')
        self.model.load_state_dict(torch.load(weight_name))
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.float()
        self.model.eval()
        self.train_data_root = Path(config.train_data_root)
        self.train_pose_root = Path(config.train_pose_root)
        
    def create_cond_img(self, resl):
        
        # 特定のサイズのフォルダを作る。        
        imsize = int(pow(2, resl))
        print(imsize)
        save_path = self.train_data_root.joinpath("pose/" + str(imsize) + '/')
        os.makedirs(save_path, exist_ok=True)
        
        # train_data_rootからファイルを取得
        p = Path(config.train_data_root + 'image')
        images = list(p.glob('*'))
        
        # 全ての画像に対してheatmapを作成
        for img_path in tqdm(images):
            img = cv2.imread(str(img_path))
            
            # TODO: データを長方形にする場合変更
            shape_dst = np.min(img.shape[:2])
            oh = (img.shape[0] - shape_dst) // 2
            ow = (img.shape[1] - shape_dst) // 2
            img = img[oh:oh+shape_dst, ow:ow+shape_dst]
            img = cv2.resize(img, (imsize, imsize))
            
            heatmap = self.get_heatmap(img) # heatmapを取得
            
            # FIXME: 名前をデータセットに依存しないように
            img_name = str(img_path).split('/')[-1].split('.')[0]
            save_name = img_name + '_' + str(imsize) + '.npy'
            np.save(save_path.joinpath(save_name), heatmap)
            
    def _get_heatmap(self, image):
        multiplier = get_multiplier(image)
        with torch.no_grad():
            _, heatmap = get_outputs(multiplier, image, self.model, 'rtpose')
        return heatmap

if __name__ == '__main__':
    cond = Condition()
    for resl in tqdm(range(2, config.max_resl)):
        cond.create_cond_img(resl)