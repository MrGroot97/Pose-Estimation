from importlib import import_module
from os import path
import cv2, os, math, json, csv
import tensorflow.lite as tflite
import numpy as np
import cv2
from matplotlib import colors
import os
from helper_function import run_inference,init_crop_region,_keypoints_and_edges_for_display,determine_crop_region,KEYPOINT_DICT
# preset declaration
from presets import Rotate
import re

# inference through tflite
interpreter = tflite.Interpreter(model_path='modals/model_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)
def movenet_tflite(image):
    # input is rgb image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (256, 256))
    # image = np.expand_dims(image, axis=0).astype("float32")
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def create_dir(path):
    if not os.path.exists(path):
            os.makedirs(path)

keypoints = {"Hip" : 0.107, "Ankle" : 0.089, "Knee" : 0.087, "Shoulder" : 0.079, "Elbow" : 0.072, "Wrist" : 0.062, "Ear": 0.035, "Nose": 0.026, "Eye" : 0.025}

if __name__ == "__main__":

    out_dir  = "output_08july"
    create_dir(out_dir)
    video_dir = "videos/original_video_data"
    preset_dir = "videos/Video Data with Presets"
    compare_dir = "videos/compare"
    preset_out_dir = "videos/preset_video_landmarks"
    unequal_landmarks = "videos/unequal_observation"

    # rows = []
    # rows.append(["Preset","Preset Val","PCK", "PCKH", "OKS","unequal_keypoint"])
    # preset_dict={'con':'contrast','exp':'exposure','rot':'rotation',
    #             'sat':'saturation','temp':'temperature','tint':'tint'}

    for dir_path in sorted(os.listdir(preset_dir)):
        if os.path.exists(os.path.join(out_dir,dir_path+".csv")):
            continue
        rows = []
        rows.append(["Preset","Preset Val","PCK", "PCKH", "OKS",'error'])
        if dir_path.endswith(".DS_Store") or (dir_path.startswith("IMG")):
            continue
        if not (dir_path.startswith("Seal_Stretch")):
            print("111111",dir_path)
            continue
        print("222222222",dir_path)
        for p_path in sorted(os.listdir(os.path.join(preset_dir,dir_path))):
            if p_path.endswith(".DS_Store"):
                continue
            # if p_path.split("_")[-2] != 'Rotation':
            #     continue
            print("33333333",p_path)
            preset_vid = cv2.VideoCapture(os.path.join(preset_dir,dir_path,p_path))

            preset_name = p_path.split("_")[-2]
            preset_val = p_path.split("_")[-1].split(".mp4")[0]
            print("preset video path is {} ,preset name is {} and preset val is {}".format(p_path,preset_name,preset_val))

            orig_vid = cv2.VideoCapture(os.path.join(video_dir,dir_path,"_".join(p_path.split("_")[:-2])+"_original.mov"))
            
            
            frame_width = int(preset_vid.get(cv2.CAP_PROP_FRAME_WIDTH))   
            frame_height = int(preset_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(preset_vid.get(cv2.CAP_PROP_FRAME_COUNT))

            crop_region = init_crop_region(frame_height, frame_width)
            preset_crop_region = init_crop_region(frame_height, frame_width)

            # method to find head size and torse height
            for jdx in range(6):
                success, frame = orig_vid.read()
                if jdx ==5:
                    pred_frame = frame.copy()
                    if preset_name == "Rotation":
                        pred_frame = Rotate(pred_frame,int(preset_val))
                        pred_frame = cv2.resize(pred_frame,(frame_width,frame_height),interpolation= cv2.INTER_AREA)
                    else:
                        pred_frame = cv2.resize(pred_frame,(frame_width,frame_height),interpolation= cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)
                    keypoints_with_scoresTemp = run_inference(movenet_tflite, frame_rgb, crop_region,crop_size=[256, 256])
                    (keypoint_locsTemp, keypoint_edgesTemp,edge_colorsTemp) = _keypoints_and_edges_for_display(keypoints_with_scoresTemp, frame_height, frame_width)
                    crop_region = determine_crop_region(keypoints_with_scoresTemp, frame_height, frame_width)
                    left_shoulder = keypoint_locsTemp[KEYPOINT_DICT["left_shoulder"]]
                    right_hip = keypoint_locsTemp[KEYPOINT_DICT["right_hip"]]
                    torso_height = math.dist(left_shoulder, right_hip)

                    left_ear = keypoint_locsTemp[KEYPOINT_DICT["left_ear"]]
                    right_ear = keypoint_locsTemp[KEYPOINT_DICT["right_ear"]]
                    head_size = math.dist(left_ear, right_ear)

                    if (head_size < 30):
                        nose = keypoint_locsTemp[KEYPOINT_DICT["nose"]]
                        # print("keypoint_locsTemp .... ",keypoint_locsTemp)
                        if (keypoints_with_scoresTemp[0, 0, KEYPOINT_DICT['left_ear'], 2] > 0.5):
                            head_size = 1.1*math.dist(left_ear, nose)  
                        else:
                            head_size = 1.1*math.dist(right_ear, nose)
                    print("head_size",head_size)

            total_keypoints = 0
            correct_keypoints_pck = 0
            correct_keypoints_pckh = 0
            total_keypoints_oks = 0
            correct_keypoints_oks = 0
            avg_dist =[]
            # not_detected_same_no_keypoints =0

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            create_dir(os.path.join(compare_dir,dir_path))
            create_dir(os.path.join(preset_out_dir,dir_path))
            create_dir(os.path.join(unequal_landmarks,dir_path,p_path.split(".mp4")[0]))
            compare_out = cv2.VideoWriter(os.path.join(compare_dir,dir_path,p_path),fourcc, 30.0, (frame_width*2,frame_height))
            preset_out = cv2.VideoWriter(os.path.join(preset_out_dir,dir_path,p_path),fourcc, 30.0, (frame_width,frame_height))
            
            print("total number of frames ", int(orig_vid.get(cv2.CAP_PROP_FRAME_COUNT)),int(preset_vid.get(cv2.CAP_PROP_FRAME_COUNT)))
            for idx in range(total_frames):
                orig_vid.set(1,idx)
                preset_vid.set(1,idx)
                success, frame = orig_vid.read()
                successP, frameP = preset_vid.read()
                # print("frame and frameP 111111",frame.shape,frameP.shape)
                if (not success) or (not successP):
                    print("\n")
                    print("idx is ",idx)
                    print("status is ",success,successP)
                    # print(orig_vid.get(cv2.CV_CAP_PROP_POS_FRAMES),preset_vid.get(cv2.CV_CAP_PROP_POS_FRAMES))
                    print(" 111111 error  in video")
                    continue
                pred_frame = frame.copy()

                if preset_name == "Rotation":
                    pred_frame = Rotate(pred_frame,-int(preset_val))
                    pred_frame = cv2.resize(pred_frame,(frame_width,frame_height),interpolation= cv2.INTER_AREA)
                    # cv2.imshow("pred_frame",pred_frame)
                else:
                    pred_frame = cv2.resize(pred_frame,(frame_width,frame_height),interpolation= cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(pred_frame.copy(), cv2.COLOR_BGR2RGB)
                keypoints_with_scores = run_inference(movenet_tflite, frame_rgb, crop_region,crop_size=[256, 256])
                (keypoint_locs, keypoint_edges,edge_colors) = _keypoints_and_edges_for_display(keypoints_with_scores, frame_height, frame_width)
                crop_region = determine_crop_region(keypoints_with_scores, frame_height, frame_width)

                frame_normal = cv2.resize(pred_frame.copy(),(frame_width,frame_height),interpolation= cv2.INTER_AREA)
                for i in range(len(keypoint_locs)):
                    cv2.circle(frame_normal, center=tuple([int(x) for x in keypoint_locs[i]]), radius=2, color=(0, 255, 255), thickness=5)
                for i in range(len(keypoint_edges)):
                    color = colors.to_rgba(edge_colors[i])[:-1]
                    cv2.line(frame_normal, tuple([int(x) for x in keypoint_edges[i][0]]), tuple([int(x) for x in keypoint_edges[i][1]]), color=tuple([int(x*255) for x in color]), thickness=3)
                
                #after preset prediction production
                preset_frame = frameP.copy()
                preset_rgb = cv2.cvtColor(preset_frame, cv2.COLOR_BGR2RGB)
                keypoints_with_scoresP = run_inference(movenet_tflite, preset_rgb, preset_crop_region,crop_size=[256, 256])
                (keypoint_locsP, keypoint_edgesP,edge_colorsP) = _keypoints_and_edges_for_display(keypoints_with_scoresP, frame_height, frame_width)
                preset_crop_region = determine_crop_region(keypoints_with_scoresP, frame_height, frame_width)
                
                frame_preset = frameP.copy()
                for i in range(len(keypoint_locsP)):
                    cv2.circle(frame_preset, center=tuple([int(x) for x in keypoint_locsP[i]]), radius=2, color=(0, 255, 255), thickness=5)
                for i in range(len(keypoint_edgesP)):
                    color = colors.to_rgba(edge_colorsP[i])[:-1]
                    cv2.line(frame_preset, tuple([int(x) for x in keypoint_edgesP[i][0]]), tuple([int(x) for x in keypoint_edgesP[i][1]]), color=tuple([int(x*255) for x in color]), thickness=3)

                result = np.hstack((frame_normal,frame_preset))

                compare_out.write(result)
                preset_out.write(frame_preset)
                cv2.imshow('movenet', result)
                if cv2.waitKey(3) & 0xFF == 27:
                    break
                # torso_height = 197.9001441743817
                # head_size = 45.85169812685796
                
                # if len(keypoint_locs) != len(keypoint_locsP):
                #     not_detected_same_no_keypoints+=1
                #     cv2.imwrite(os.path.join(unequal_landmarks,dir_path,p_path.split(".mp4")[0],"frame_"+str(idx)+".jpg"),result)
                    
                x_vals = []
                y_vals = []
                for i in range(len(keypoint_locs)):
                    point=[int(x) for x in keypoint_locs[i]]
                    x_vals.append(point[0])
                    y_vals.append(point[1])

                x_vals.sort()
                y_vals.sort()
                width = x_vals[-1]-x_vals[0]
                height = y_vals[-1]-y_vals[0]
                scale_area = width*height
                # print("info length",len(keypoint_locs),len(keypoint_locsP))
                max_landmarks = max(len(keypoint_locs),len(keypoint_locsP))
                min_landmarks = min(len(keypoint_locs),len(keypoint_locsP))
                for i in range(len(keypoint_locs),):
                    # if i >(min_landmarks-1):
                    #     total_keypoints = total_keypoints + 1
                    #     total_keypoints_oks = total_keypoints_oks + 1
                    # else:
                    total_keypoints = total_keypoints + 1
                    gt = [int(x) for x in keypoint_locs[i]]
                    pred = [int(x) for x in keypoint_locsP[i]]
                    pck = math.dist(gt, pred)
                    avg_dist.append(pck)
                    if i in [11,12]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Hip"]
                    elif i in [15,16]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Ankle"]
                    elif i in [13,14]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Knee"]
                    elif i in [5,6]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Shoulder"]
                    elif i in [7,8]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Elbow"]
                    elif i in [9,10]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Wrist"]
                    elif i in [3,4]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Ear"]
                    elif i in [0]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Nose"]
                    elif i in [1,2]:
                        total_keypoints_oks = total_keypoints_oks + 1
                        k = keypoints["Eye"]   
                    else:
                        k =0
                    if k != 0:
                        oks = math.exp(-pck/((2*scale_area)*k*k))
                        if oks > 0.75:
                            correct_keypoints_oks = correct_keypoints_oks + 1

                    if pck < 0.2*torso_height: 
                        correct_keypoints_pck = correct_keypoints_pck + 1

                    if pck < 0.5*head_size: 
                        correct_keypoints_pckh = correct_keypoints_pckh + 1

            compare_out.release()
            preset_out.release()
            cv2.destroyAllWindows()
            PCK = (correct_keypoints_pck/total_keypoints)*100
            PCKH = (correct_keypoints_pckh/total_keypoints)*100
            OKS = (correct_keypoints_oks/total_keypoints_oks)*100
            # unequal_keypoints = (not_detected_same_no_keypoints/total_frames)*100
            avg_distance = np.mean(avg_dist)
            rows.append([preset_name,preset_val, PCK, PCKH, OKS,avg_distance])
            print("printing details for ",p_path)
            print("PCK : " , PCK)
            print("PCKH : " , PCKH)
            print("OKS : " , OKS)
            print("avg dist :",avg_distance)
            # print("NO of frame which have different keypoint detection ...",unequal_keypoints)
        
        # print(rows)
        filename = os.path.join(out_dir,dir_path+".csv")
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerows(rows)