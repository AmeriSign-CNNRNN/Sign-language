import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
#videodir = "E:\\Sign Language Dataset\\Chinese SL Dataset (Sentences)\\color\\"
videodir = 'C:\\Users\\ECE-ML\\Desktop\\sample data\\'
label_dir = "E:\\Sign Language Dataset\\Chinese SL Dataset (Sentences)\\corpus.txt"

class VideoDataset(Dataset):

    def __init__(self, label_dir, videodir, channels, timeDepth, transform=None):
        total_number = 0
        data = os.listdir(videodir)
        for paths, dirs, files in os.walk(videodir):
            total_number += len(files)
        self.label_dir = label_dir
        self.total_number = total_number
        self.videodir = videodir
        self.channels = channels
        self.timeDepth = timeDepth
        self.transform = transform
        self.video_label = data

    def __len__(self):
        return self.total_number

    def video_frame_clip_list(self,videopath, timeDepth):
        cap = cv2.VideoCapture(videopath)
        frame_list = []

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_list.append(frame)
            else:
                break
        return [frame_list[i:i + timeDepth] for i in range(len(frame_list)) if i % timeDepth == 0]

    def readVideo(self,videofile, timeDepth, channels):
        i = 0
        cap = cv2.VideoCapture(videofile)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        xSize = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ySize = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_clips = int(nframe // timeDepth)
        frames = torch.FloatTensor(n_clips, timeDepth, channels, xSize, ySize)
        data = self.video_frame_clip_list(videofile, timeDepth)
        for clip in range(len(data) - 1):
            frames[clip, :, :, :, :] = clip
            for i in range(len(data[clip])):
                frame = torch.from_numpy(data[clip][i])
                frame = frame.permute(2, 1, 0)
                frames[:, i, :, :, :] = frame
        return frames

    def video_file_path(self,videodir):
        Full_video_label_path = []
        video_name_path = []
        data = os.listdir(videodir)
        for i in range(len(data)):
            video_label_path = os.path.join(videodir, data[i])
            Full_video_label_path.append(video_label_path)
        for j in range(len(Full_video_label_path)):
            for path, dirs, files in os.walk(Full_video_label_path[j]):
                for file in files:
                    video_path = os.path.join(path, file)
                    video_name_path.append(video_path)
        return video_name_path #totall video_path 20000多


    def __getitem__(self, index):
        dataset=[]
        labels=self.video_label
        for label in labels:
            for one_video in self.video_file_path(self.videodir):
                video_path_str = one_video.split("\\")
                label_in_path = video_path_str[-2]
                if label_in_path==label:
                    sample=tuple((label,self.readVideo(one_video,self.timeDepth,3)))
                    dataset.append(sample)
                else:
                    continue

        Video_label=dataset[index][0]
        Video_data=dataset[index][1]

        return Video_label, Video_data

ex=VideoDataset(label_dir,videodir,3,16)

print(ex[0])

