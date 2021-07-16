import tensorflow as tf
import tensorflow.lite as tflite
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import colors
import argparse
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
KEYPOINT_EDGE_CONNECTIONS = [
    (NOSE, RIGHT_EYE_INNER),
    (RIGHT_EYE_INNER, RIGHT_EYE),
    (RIGHT_EYE, RIGHT_EYE_OUTER),
    (RIGHT_EYE_OUTER, RIGHT_EAR),
    (NOSE, LEFT_EYE_INNER),
    (LEFT_EYE_INNER, LEFT_EYE),
    (LEFT_EYE, LEFT_EYE_OUTER),
    (LEFT_EYE_OUTER, LEFT_EAR),
    (MOUTH_RIGHT, MOUTH_LEFT),
    (RIGHT_SHOULDER, LEFT_SHOULDER),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_WRIST, RIGHT_PINKY),
    (RIGHT_WRIST, RIGHT_INDEX),
    (RIGHT_WRIST, RIGHT_THUMB),
    (RIGHT_PINKY, RIGHT_INDEX),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_PINKY),
    (LEFT_WRIST, LEFT_INDEX),
    (LEFT_WRIST, LEFT_THUMB),
    (LEFT_PINKY, LEFT_INDEX),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_HIP, LEFT_HIP),
    (RIGHT_HIP, RIGHT_KNEE),
    (LEFT_HIP, LEFT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (LEFT_ANKLE, LEFT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
]

interpreter = tflite.Interpreter(model_path='blazepose_models/saved_model_full/model_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# pp.pprint(output_details)
def movenet_tflite(image):
    # input is rgb image
    image = cv2.resize(image, (256, 256))
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))/255
    image = np.expand_dims(image, axis=0).astype("float32")
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output1 = interpreter.get_tensor(output_details[0]['index'])
    # output2 = interpreter.get_tensor(output_details[1]['index'])
    # output3 = interpreter.get_tensor(output_details[2]['index'])
    # output4 = interpreter.get_tensor(output_details[3]['index'])
    # output5 = interpreter.get_tensor(output_details[4]['index'])
    
    return output1

def convert_preds_to_xy(preds,input_image):
    preds = preds.flatten()
    kpts = []
    num_dimension = 5
    num_keypoints=39
    input_size = 256
    h,w,_ = input_image.shape
    for idx in range(0,num_keypoints*num_dimension,num_dimension):
        visibility = preds[idx+3]
        if not visibility:
            continue
        # kpt_dict={"x":int((w/input_size)*preds[idx]),"y":int((h/input_size)*preds[idx+1]),"z":preds[idx+2],"vis":preds[idx+3],"pres":preds[idx+4]}
        landmark_list = [int((w/input_size)*preds[idx]),int((h/input_size)*preds[idx+1]),preds[idx+2],preds[idx+3],preds[idx+4]]
        # pp.pprint(kpt_dict)
        kpts.append(landmark_list)

    return kpts
    
def draw_keypoints(image,keypoints):
    num_dimension = 5
    num_keypoints=len(keypoints)
    # print(".......",len(keypoints))
    for idx in range(num_keypoints):
        # for jdx in range(num_dimension):
        cv2.circle(image, center=tuple([keypoints[idx][0],keypoints[idx][1]]), radius=3, color=(0, 255, 255), thickness=5)
    
    for idx in range(len(KEYPOINT_EDGE_CONNECTIONS)):
        # print(KEYPOINT_EDGE_CONNECTIONS[idx])
        p1_idx = KEYPOINT_EDGE_CONNECTIONS[idx][0]
        p2_idx = KEYPOINT_EDGE_CONNECTIONS[idx][1]
        p1 = [keypoints[p1_idx][0],keypoints[p1_idx][1]]
        p2 = [keypoints[p2_idx][0],keypoints[p2_idx][1]]
        cv2.line(frame, p1, p2, color=tuple([0,255,0]),thickness=3)

    return image

def create_dir(path):
    if not os.path.exists(path):
            os.makedirs(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo script for human segmentation')
    parser.add_argument("--vid",default='video',help = 'Path to input video directory')
    parser.add_argument("--out_dir",default='outputs',help = 'Path to output directory')
    # parser.add_argument("--model_name",default='movenet_thunder',help = 'movenet model movenet_lightning/movenet_thunder')
    args = parser.parse_args()

    out_dir  = args.out_dir
    create_dir(out_dir)
    video_dir = args.vid

    for vid_path in os.listdir(video_dir):
        if vid_path.endswith(".DS_Store"):
            continue
        
        vidObj = cv2.VideoCapture(os.path.join(video_dir,vid_path))
        # while vidObj.isOpened():
        frame_width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))   
        frame_height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

        print("processing video and total frame",vid_path,total_frames)
        # video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(out_dir,os.path.basename(vid_path).split('.')[0] + '.mp4'),fourcc, 30.0, (frame_width,frame_height))

        for idx in range(total_frames):
            success, frame = vidObj.read()
            if not success:
                continue
            print("processing frame .... ",idx)
            # print("input size and model name ",input_size,model_name)
            # frame = cv2.resize(frame, (256, 256))
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype("float32")
            results = movenet_tflite(frame)
            keypoints = convert_preds_to_xy(results,frame)
            
            modified_frame = draw_keypoints(frame,keypoints)

            cv2.imshow('BlazePose', modified_frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break