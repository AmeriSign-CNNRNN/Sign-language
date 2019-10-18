import torch
import cv2
import os
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
videodir = 'C:\\Users\\ECE-ML\\Desktop\\sample video data\\'
label_dir = "D:\\Sign Language Dataset\\Chinese SL Dataset (Words)\\dictionary.txt"

class VideoDataset(Dataset):

    def __init__(self, label_dir, videodir, channels, timeDepth, transform=None):
        total_number = 0
        video_label = os.listdir(videodir)
        for paths, dirs, files in os.walk(videodir):
            total_number += len(files)
        self.label_dir = label_dir
        self.total_number = total_number
        self.videodir = videodir
        self.channels = channels
        self.timeDepth = timeDepth
        self.transform = transform
        self.video_label = video_label

    def __len__(self):
        return self.total_number

    def video_file_path(self):
        Full_video_label_path = []
        for single_dir in self.video_label:
            video_name_path = []
            data_full_dir = os.path.join(self.videodir, single_dir)
            for video_file in os.listdir(data_full_dir):
                video_file_path = os.path.join(data_full_dir, video_file)
                video_name_path.append(video_file_path)
            Full_video_label_path.append(video_name_path)
        return Full_video_label_path

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



    def __getitem__(self, index):
        dataset=[]
        for num_label in range(len(self.video_label)):
            label=self.video_label[num_label]
            video_path_in_one_label=self.video_file_path()[num_label]
            for video_file_path in video_path_in_one_label:
                frame=self.readVideo(video_file_path,16,3)
                label_and_frame=tuple((label,frame))
                dataset.append(label_and_frame)

        Video_label=dataset[index][0]
        Video_data=dataset[index][1]

        return Video_label, Video_data

ex=VideoDataset(label_dir,videodir,3,16)
