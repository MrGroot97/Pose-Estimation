from importlib import import_module
import cv2, os, math, json, csv
import tensorflow.lite as tflite
import numpy as np
import cv2
from matplotlib import colors
import os
# from helper_function import run_inference,init_crop_region,_keypoints_and_edges_for_display,determine_crop_region,KEYPOINT_DICT
import helper_function


# inference through tflite
interpreter = tflite.Interpreter(model_path='modals/model_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def movenet_tflite(image):
    # input is rgb image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (256, 256))
    # image = np.expand_dims(image, axis=0).astype("float32")
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


keypoints = {"Hip" : 0.107, "Ankle" : 0.089, "Knee" : 0.087, "Shoulder" : 0.079, "Elbow" : 0.072, "Wrist" : 0.062, "Ear": 0.035, "Nose": 0.026, "Eye" : 0.025}

if __name__ == "__main__":

    out_dir  = "hingeHealthOutput"
    helper_function.create_dir(out_dir)
    image_dir = "HingeHealth_data/HingeHealth_original_data"

    rows = []
    rows.append(["Image Name","PCK", "PCKH", "OKS","ERROR"])
    total_keypoints = 0
    correct_keypoints_pck = 0
    correct_keypoints_pckh = 0
    total_keypoints_oks = 0
    correct_keypoints_oks = 0
    avg_dist =[]
    
    for img_path in sorted(os.listdir(image_dir)):
        
        if not img_path.endswith(".jpg") or ( not img_path.endswith(".jpeg") or img_path.endswith(".png")):
            continue
            
            orgimglist = [x for x in sorted(os.listdir(os.path.join(video_dir,dir_path,p_dir))) if (not x.endswith(".DS_Store"))]
            preset_imglist = [x for x in sorted(os.listdir(os.path.join(preset_dir,dir_path,p_dir))) if (not x.endswith(".DS_Store"))]
            
            for i in range(1):
                image = cv2.imread(os.path.join(video_dir,dir_path,p_dir,orgimglist[0]))
                # print(os.path.join(video_dir,dir_path,p_dir,orgimglist[0]))
                frame_width = image.shape[1]
                frame_height = image.shape[0]
                # minising crop region
                crop_region = init_crop_region(frame_height, frame_width)
                #print("initial crop region",crop_region)
                psedoframe = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                key_scores = run_inference(movenet, psedoframe, crop_region,crop_size=[256, 256])
                # not needed here
                (keypoint_locsp, keypoint_edgesp,edge_colorsp) = _keypoints_and_edges_for_display(key_scores, frame_height, frame_width)
                crop_region = determine_crop_region(key_scores, frame_height, frame_width)
                preset_crop_region= crop_region.copy()
                for i in range(len(keypoint_locsp)):
                    cv2.circle(image, center=tuple([int(x) for x in keypoint_locsp[i]]), radius=2, color=(0, 255, 255), thickness=5)
                for i in range(len(keypoint_edgesp)):
                    color = colors.to_rgba(edge_colorsp[i])[:-1]
                    cv2.line(image, tuple([int(x) for x in keypoint_edgesp[i][0]]), tuple([int(x) for x in keypoint_edgesp[i][1]]), color=tuple([int(x*255) for x in color]), thickness=3)

                # cv2.imshow('image', cv2.resize(image,(500,1000)))
                # if cv2.waitKey(0) & 0xFF == 27:
                #     cv2.destroyAllWindows()
            #print("final crop region",crop_region,preset_crop_region)
            
            # method to find head size and torse height
            for i in range(1):
                frame_head = cv2.imread(os.path.join(video_dir,dir_path,p_dir,orgimglist[0]))
                frame_head_rgb = cv2.cvtColor(frame_head.copy(), cv2.COLOR_BGR2RGB)
                keyScores_head = run_inference(movenet, frame_head_rgb, crop_region,crop_size=[256, 256])
                (keypoint_locsHead, keypoint_edgesHead,edge_colorsHead) = _keypoints_and_edges_for_display(keyScores_head, frame_height, frame_width)

                left_shoulder = keypoint_locsHead[KEYPOINT_DICT["left_shoulder"]]
                right_hip = keypoint_locsHead[KEYPOINT_DICT["right_hip"]]
                torso_height = math.dist(left_shoulder, right_hip)

                left_ear = keypoint_locsHead[KEYPOINT_DICT["left_ear"]]
                right_ear = keypoint_locsHead[KEYPOINT_DICT["right_ear"]]
                head_size = math.dist(left_ear, right_ear)

                if (head_size < 30):
                    nose = keypoint_locsHead[KEYPOINT_DICT["nose"]]
                    if (keyScores_head[0, 0, KEYPOINT_DICT['left_ear'], 2] > 0.5):
                        head_size = 1.1*math.dist(left_ear, nose)  
                    else:
                        head_size = 1.1*math.dist(right_ear, nose)
                print("head_size ...........",head_size)
            # head_size = 0
            # torso_height = 0
            # p_list = []
            # o_list = []
            preset_list = [x for x in preset_imglist],[x for x in preset_imglist],[x for x in preset_imglist]

            for idx in range(len(preset_imglist)):
                fralcount+=1
                preset_img = cv2.imread(os.path.join(preset_dir,dir_path,p_dir,preset_imglist[idx]))
                preset_name = preset_imglist[idx].split("_")[-2]
                preset_val = preset_imglist[idx].split("_")[-1].split(".jpg")[0]
                # print("preset video path is {} ,preset name is {} and preset val is {}".format(preset_imglist[idx],preset_name,preset_val))
                # print("original name ","_".join(preset_imglist[idx].split("_")[:2])+".jpg")
                original_img = cv2.imread(os.path.join(video_dir,dir_path,p_dir,"_".join(preset_imglist[idx].split("_")[:2])+".jpg"))
                frame_width = preset_img.shape[1] 
                frame_height = preset_img.shape[0]

                org_frame_width = original_img.shape[1] 
                org_frame_height = original_img.shape[0]
                #print("some info .... ",preset_img.shape,original_img.shape)
            
        
                create_dir(os.path.join(compare_dir,dir_path,p_dir))
                create_dir(os.path.join(preset_out_dir,dir_path,p_dir))

                # pred_frame = cv2.resize(original_img.copy(),(frame_width,frame_height),interpolation= cv2.INTER_AREA)
                #cv2.imshow('pred_frame', cv2.resize(pred_frame,(1080,540)))
                frame_rgb = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB)
                keypoints_with_scores = run_inference(movenet, frame_rgb, crop_region,crop_size=[256, 256])
                (keypoint_locs, keypoint_edges,edge_colors) = _keypoints_and_edges_for_display(keypoints_with_scores, org_frame_height, org_frame_width)
                # if preset_imglist[idx].split("_")[0]+".jpg" in o_list:
                #     print("")
                # else:
                #     o_list.append(preset_imglist[idx].split("_")[0]+".jpg")
                crop_region = determine_crop_region(keypoints_with_scores, org_frame_height, org_frame_width)

                #frame_normal = cv2.resize(original_img.copy(),(frame_width,frame_height),interpolation= cv2.INTER_AREA)
                frame_normal = original_img.copy()
                for i in range(len(keypoint_locs)):
                    cv2.circle(frame_normal, center=tuple([int(x) for x in keypoint_locs[i]]), radius=2, color=(0, 255, 255), thickness=5)
                for i in range(len(keypoint_edges)):
                    color = colors.to_rgba(edge_colors[i])[:-1]
                    cv2.line(frame_normal, tuple([int(x) for x in keypoint_edges[i][0]]), tuple([int(x) for x in keypoint_edges[i][1]]), color=tuple([int(x*255) for x in color]), thickness=3)
                
                #after preset prediction production
                preset_frame = preset_img.copy()
                #cv2.imshow('preset_frame', cv2.resize(preset_frame,(1080,540)))
                preset_rgb = cv2.cvtColor(preset_frame, cv2.COLOR_BGR2RGB)
                keypoints_with_scoresP = run_inference(movenet, preset_rgb, preset_crop_region,crop_size=[256, 256])
                (keypoint_locsP, keypoint_edgesP,edge_colorsP) = _keypoints_and_edges_for_display(keypoints_with_scoresP, frame_height, frame_width)
                preset_crop_region = determine_crop_region(keypoints_with_scoresP, frame_height, frame_width)
                
                frame_preset = preset_img.copy()
                for i in range(len(keypoint_locsP)):
                    cv2.circle(frame_preset, center=tuple([int(x) for x in keypoint_locsP[i]]), radius=2, color=(0, 255, 255), thickness=5)
                for i in range(len(keypoint_edgesP)):
                    color = colors.to_rgba(edge_colorsP[i])[:-1]
                    cv2.line(frame_preset, tuple([int(x) for x in keypoint_edgesP[i][0]]), tuple([int(x) for x in keypoint_edgesP[i][1]]), color=tuple([int(x*255) for x in color]), thickness=3)

                result = np.hstack((frame_normal,frame_preset))
            

                # cv2.imwrite(os.path.join(compare_dir,dir_path,p_dir,preset_imglist[idx]),result)
                # cv2.imwrite(os.path.join(preset_out_dir,dir_path,p_dir,preset_imglist[idx]),frame_preset)
                cv2.imshow('movenet', cv2.resize(result,(1080,540)))
                if cv2.waitKey(150) & 0xFF == 27:
                    break
                    
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
                # max_landmarks = max(len(keypoint_locs),len(keypoint_locsP))
                # min_landmarks = min(len(keypoint_locs),len(keypoint_locsP))
                pckhcount =0
                for i in range(len(keypoint_locs)):
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
                        #print("yes")
                        pckhcount+=1
                        correct_keypoints_pckh = correct_keypoints_pckh + 1
                if pckhcount< 15:
                    cv2.imwrite(os.path.join(compare_dir,dir_path,p_dir,preset_imglist[idx]),result)
                    cv2.imwrite(os.path.join(preset_out_dir,dir_path,p_dir,preset_imglist[idx]),frame_preset)
                    # print("\n pck is ......111111,.....",pck,0.5*head_size,head_size)
                    # print("correct_keypoints_pckh .... ",correct_keypoints_pckh)
                    # print("total_keypoints .... ",total_keypoints)
                
                cv2.imwrite(os.path.join(compare_dir,dir_path,p_dir,preset_imglist[idx]),result)
                cv2.imwrite(os.path.join(preset_out_dir,dir_path,p_dir,preset_imglist[idx]),frame_preset)

            cv2.destroyAllWindows()
            PCK = (correct_keypoints_pck/total_keypoints)*100
            PCKH = (correct_keypoints_pckh/total_keypoints)*100
            OKS = (correct_keypoints_oks/total_keypoints_oks)*100
            avg_distance = np.mean(avg_dist)
            rows.append([p_dir, PCK, PCKH, OKS,avg_distance])
            print("printing details for ",p_dir,fralcount,total_keypoints)
            print("PCK : " , PCK)
            print("PCKH : " , PCKH)
            print("OKS : " , OKS)
            print("avg dist :",avg_distance)
        
        filename = os.path.join(out_dir,dir_path+".csv")
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerows(rows)