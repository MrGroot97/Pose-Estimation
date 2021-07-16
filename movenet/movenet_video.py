import tensorflow as tf
import tensorflow.lite as tflite
import numpy as np
import cv2
import argparse
import os
from utils import  helper_function

if __name__ == "__main__":
    # sample run command
    # python movenet_video.py --video ../../movenet_evaluation/IdleAvoiderPortrait/Donkey\ Kicks.mp4 --out_dir testing --model_name thunder
    parser = argparse.ArgumentParser(description='Demo script for movenet pose estimation')
    parser.add_argument("--video",default=0,help = 'Path to input video file',help="input video path")
    parser.add_argument("--out_dir",default='outputs',help = 'Path to output directory')
    parser.add_argument("--model_name",default='movenet_thunder',help = 'movenet model movenet_lightning/movenet_thunder')
    args = parser.parse_args()

    out_dir  = args.out_dir
    helper_function.create_dir(out_dir)
    video_path = args.video
    model_name = args.model_name
    if "lightning" in model_name:
        input_size = 192
        interpreter = tflite.Interpreter(model_path='models/movenet_singlepose_lightning_3.tflite', num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    elif "thunder" in model_name:
        input_size = 256
        interpreter = tflite.Interpreter(model_path='models/movenet_singlepose_thunder_3.tflite', num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        raise ValueError("Unsupported model name: %s" % model_name)
    
    if not (video_path.split(".")[-1] in helper_function.SUPPORTED_VIDEO_EXTENSIONS):
        raise ValueError("Unsupported video type: %s" % video_path)
        
    vidObj = cv2.VideoCapture(os.path.join(video_path))
    frame_width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))   
    frame_height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(out_dir,os.path.basename(video_path).split('.')[0] + '.mp4'),fourcc, 30.0, (frame_width,frame_height))

    crop_region = helper_function.init_crop_region(frame_height, frame_width)
    for idx in range(total_frames):
        success, frame = vidObj.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_with_scores = helper_function.run_inference(helper_function.movenet_tflite,interpreter,input_details,output_details, frame_rgb, crop_region,crop_size=[input_size, input_size])
        
        (keypoint_locs, keypoint_edges,edge_colors) = helper_function._keypoints_and_edges_for_display(keypoints_with_scores, frame_height, frame_width)
        crop_region = helper_function.determine_crop_region(keypoints_with_scores, frame_height, frame_width)

        frame = helper_function.draw_prediction_on_image_cv(frame,keypoint_locs, keypoint_edges,edge_colors)

        cv2.imshow('movenet', frame)
        if cv2.waitKey(3) & 0xFF == 27:
            break

        out.write(frame)
    vidObj.release()
    out.release()
