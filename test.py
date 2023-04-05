import unittest
import os
import numpy as np
import cv2
from model_prediction import predict_on_video

class TestPredictOnVideo(unittest.TestCase):

    def setUp(self):
        #define the location
        self.video_file_path = "Testdata/u.mp4"
        self.output_file_path = "Testdata/test_output.mp4"
        self.detect_video = "Testdata/test_detect.mp4"
        self.SEQUENCE_LENGTH = 30

        # create a test video with random frames
        self.width, self.height = 640, 480
        self.fps = 30
        self.frames_num = 50
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_file_path, fourcc, self.fps, (self.width, self.height))
        for i in range(self.frames_num):
            frame = np.random.randint(0, 256, (self.height, self.width, 3)).astype(np.uint8)
            self.video_writer.write(frame)
        self.video_writer.release()

    def tearDown(self):
        # delete the test video and output files
        os.remove(self.video_file_path)
        os.remove(self.output_file_path)
        os.remove(self.detect_video)

    def test_predict_on_video(self):
        predict_on_video(self.video_file_path, self.output_file_path, self.detect_video, self.SEQUENCE_LENGTH)
        # check if output file is created
        self.assertTrue(os.path.isfile(self.output_file_path))
        # check if detected video file is created
        self.assertTrue(os.path.isfile(self.detect_video))
        # check if the output video has the same number of frames as the input video
        input_video = cv2.VideoCapture(self.video_file_path)
        output_video = cv2.VideoCapture(self.output_file_path)
        self.assertEqual(input_video.get(cv2.CAP_PROP_FRAME_COUNT), output_video.get(cv2.CAP_PROP_FRAME_COUNT))
        input_video.release()
        output_video.release()

if __name__ == '__main__':
    unittest.main()
