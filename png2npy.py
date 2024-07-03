import cv2
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

def extract_video_features(video_path, out_path, idx, img_transform):
    video_list = []
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = img_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        video_list.append(frame)
    video_clip = torch.cat(video_list, axis=0)
    out_path = out_path + '/' + str(idx) + '.npy'
    np.save(out_path, video_clip)
    return


def Transform(img):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    img = transform(img)
    return img

if __name__ == '__main__':

    all_path = r"/public_bme/data/v-lijm/REACT_2024/test/Video_files"
    out_all_path = r"/public_bme/data/v-lijm/REACT_2024/video_data"  ## 将图片先进行数据处理，再打包成npy数据

    path_2 = os.listdir(all_path) # NoXI ..
    for i2 in path_2:
        t2 = os.path.join(all_path, i2)  ## ./NoXI
        path_3 = os.listdir(t2)
        for i3 in tqdm(path_3):
            t3 = os.path.join(t2, i3)  ## ./NoXI/001_2016-03-17_Pairs
            path_4 = os.listdir(t3)
            for i4 in path_4:
                t4 = os.path.join(t3, i4)  ## ./NoXI/001_2016-03-17_Pairs/Expert_video
                path_5 = os.listdir(t4)
                for i5 in path_5:
                    t5 = os.path.join(t4, i5)  ## ./NoXI/001_2016-03-17_Pairs/Expert_video/1.mp4
                    print(t5)
                    video_path = t5
                    out_path = out_all_path + '/' + i2 + '/' + i3 + '/' + i4
                    os.makedirs(out_path, exist_ok=True)
                    ## ./NoXI/001_2016-03-17_Pairs/Expert_video
                    extract_video_features(video_path, out_path, i5[:-4], Transform)

