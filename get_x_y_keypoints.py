import json
import os

path='C:\\Users\\ECE-ML\\Desktop\\sample data\\'

class keypoint:
    def __init__(self, path):
        self.path = path

    def get_jsonfile_path(self):
        keypoint_dirs = []
        all_label_jsonfile = []
        for dir in os.listdir(self.path):
            keypoint_dir = os.path.join(self.path, dir)
            keypoint_dirs.append(keypoint_dir)  # get keypoint path
        for path_keypoint_dir in keypoint_dirs:  # all of the keypoint path
            same_label_jsonfile = []
            for json_file in os.listdir(path_keypoint_dir):  # all the file in path
                json_file_dir = os.path.join(path_keypoint_dir, json_file)  # join the path and name of file
                same_label_jsonfile.append(json_file_dir)
            all_label_jsonfile.append(same_label_jsonfile)
        return all_label_jsonfile
        # print(json_file_dir)
        # C:\Users\ECE-ML\Desktop\video_json_word\000000\P01_01_00_0_color.json

    def get_x_y(self, keypoint):
        allvideo_coordinate = []
        for onelabel_folder in self.get_jsonfile_path():
            onelabel_allvideo_coordinate = []
            for onejsonfile in onelabel_folder:
                video_frame_coordinate = []
                with open(onejsonfile, 'r') as f:
                    data = json.load(f)  # all frames data in one video
                    for num_frame in range(1, len(data), 1):
                        # frame number in one video
                        frame_coordinate = []
                        frame_data = data[num_frame]
                        data_body = frame_data[keypoint]
                        data_body_list = data_body[0]
                        for data_x_y_num in range(len(data_body_list)):  # one frame x and y
                            data_keypoint = data_body_list[data_x_y_num]
                            data_x_y = tuple((data_keypoint[0], data_keypoint[1]))
                            frame_coordinate.append(data_x_y)
                        video_frame_coordinate.append(frame_coordinate)
                onelabel_allvideo_coordinate.append(video_frame_coordinate)
            allvideo_coordinate.append(onelabel_allvideo_coordinate)
        return allvideo_coordinate
    def left_hand_keypoint(self):
        return self.get_x_y('Left hand keypoint')

    def right_hand_keypoint(self):
        return self.get_x_y('right hand keypoint')

    def body_keypoint(self):
        return self.get_x_y('body keypoint')

    def Face_keypoint(self):
        return self.get_x_y('Face keypoint')



