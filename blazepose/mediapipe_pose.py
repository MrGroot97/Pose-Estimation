import cv2, os, math, json, csv
from PIL import Image
import imageio
import numpy as np

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
'''
path = 'videos/'
out_path = "new_data_frames/"
for vid in os.listdir(path):
    if vid != ".DS_Store":
        print(vid)
        cap = cv2.VideoCapture(os.path.join(path,vid))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("frames", frames)
        for frame in range(frames):
            success, image = cap.read()
            if not success:
                print("in not success")
                continue
            fname = vid[:-4]
            if os.path.isdir(os.path.join(out_path,fname)) is False:
                os.mkdir(os.path.join(out_path,fname))
            name = os.path.join(out_path,fname)
            cv2.imwrite(os.path.join(os.path.join(name,str(frame)+'.jpg')), image)      
        cap.release()


'''
# For static images:
keypoints = {"Hip" : 0.107, "Ankle" : 0.089, "Knee" : 0.087, "Shoulder" : 0.079, "Elbow" : 0.072, "Wrist" : 0.062, "Ear": 0.035, "Nose": 0.026, "Eye" : 0.025}
preset_vid_path = 'new_data_preset_videos/'
for preset in os.listdir(preset_vid_path):
    if preset != ".DS_Store":
        rows = []
        rows.append(["Preset", "PCK", "PCKH", "OKS"])
        # original_folder_name = ''
        # x = preset.split("_")
        # for i in range(len(x)-2):
            # original_folder_name = original_folder_name + '_' + x[i]
        # original_folder_name = original_folder_name[1:]

        print("new_data_frames/"+preset+"/0.jpg")
        frame_1 = cv2.imread("new_data_frames/"+preset+"/0.jpg")
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5) as pose:
            image_height, image_width, _ = frame_1.shape
            result_f1 = pose.process(cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB))
            left_shoulder = [result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x*image_width, result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y*image_height]
            right_hip = [result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x*image_width, result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y*image_height]
            torso_height = math.dist(left_shoulder, right_hip)
            print(torso_height)

            left_ear = [result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x*image_width, result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y*image_height]
            right_ear = [result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x*image_width, result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y*image_height]
            print("Left ", left_ear, result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].visibility)
            print("RIGHT_EAR ", right_ear, result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].visibility)
            head_size = math.dist(left_ear, right_ear)
            
            if (head_size < 20):
            #     head_size = math.dist(left_ear, right_ear)
            # else:
                nose = [result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x*image_width, result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y*image_height]
                if (result_f1.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].visibility > 0.5):
                    head_size = 1.1*math.dist(left_ear, nose)  
                else:
                     head_size = 1.1*math.dist(right_ear, nose)
            print(head_size)
        
            for video in os.listdir(os.path.join(preset_vid_path,preset)):
                if video != ".DS_Store":
                    total_keypoints = 0
                    correct_keypoints_pck = 0
                    correct_keypoints_pckh = 0
                    total_keypoints_oks = 0
                    correct_keypoints_oks = 0
                    print(os.path.join(preset_vid_path+"/"+preset, video))
                    cap = cv2.VideoCapture(os.path.join(preset_vid_path+"/"+preset, video))
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) 

                    write_to = 'new_data_preset_outputs/'+video
                    writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=fps)
                    with mp_pose.Pose(
                        static_image_mode=True,
                        model_complexity=2,
                        min_detection_confidence=0.5) as pose:
                        for frame in range(frames):
                            success, aug_image = cap.read() 
                            try:

                                print(os.path.join('new_data_frames/'+preset,str(frame)+'.jpg'))
                                image = cv2.imread(os.path.join('new_data_frames/'+preset,str(frame)+'.jpg'))
                                image_height, image_width, _ = image.shape
                                # Convert the BGR image to RGB before processing.
                                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                                aug_results = pose.process(cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))
                                if not aug_results.pose_landmarks:
                                    total_keypoints = total_keypoints + 33
                                    total_keypoints_oks = total_keypoints_oks + 21
                                    continue
                                annotated_image = aug_image.copy()
                                mp_drawing.draw_landmarks(
                                    annotated_image, aug_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                                writer.append_data(np.asarray(annotated_image))

                                x_vals = []
                                y_vals = []
                                for i in mp_holistic.PoseLandmark:
                                    x_vals.append(results.pose_landmarks.landmark[i].x*image_width)
                                    y_vals.append(results.pose_landmarks.landmark[i].y*image_height)

                                x_vals.sort()
                                y_vals.sort()
                                width = x_vals[-1]-x_vals[0]
                                height = y_vals[-1]-y_vals[0]
                                # print(width, height)
                                scale_area = width*height

                                
                                for i in mp_holistic.PoseLandmark:
                                    total_keypoints = total_keypoints + 1
                                    gt = [results.pose_landmarks.landmark[i].x*image_width, results.pose_landmarks.landmark[i].y*image_height]
                                    pred = [aug_results.pose_landmarks.landmark[i].x*image_width, aug_results.pose_landmarks.landmark[i].y*image_height]
                                    pck = math.dist(gt, pred)
                                    # print("dist : ", pck)
                                    if "HIP" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Hip"]

                                    elif "ANKLE" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Ankle"]
                                    elif "KNEE" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Knee"]
                                    elif "SHOULDER" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Shoulder"]
                                    elif "ELBOW" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Elbow"]
                                    elif "WRIST" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Wrist"]
                                    elif "EAR" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Ear"]
                                    elif "NOSE" in str(i):
                                        total_keypoints_oks = total_keypoints_oks + 1
                                        k = keypoints["Nose"]
                                    elif "EYE" in str(i):
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
                                
                            except Exception as e:
                                print("error ::", preset, e)

                    PCK = (correct_keypoints_pck/total_keypoints)*100
                    PCKH = (correct_keypoints_pckh/total_keypoints)*100
                    OKS = (correct_keypoints_oks/total_keypoints_oks)*100
                    rows.append([video[:-4], PCK, PCKH, OKS])
                    print("PCK : " , (correct_keypoints_pck/total_keypoints)*100)
                    print("PCKH : " , (correct_keypoints_pckh/total_keypoints)*100)
                    print("OKS : " , (correct_keypoints_oks/total_keypoints_oks)*100)
                    writer.close()
            
        print(rows)
        filename = "new_data_preset_csv/"+preset+".csv"
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerows(rows)

'''
# For webcam input:
cap = cv2.VideoCapture("yoga.mp4")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  idx = 0
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    cv2.imwrite('data/frames/frame_' + str(idx).zfill(6) + '.png', image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite('output/marker/frame_' + str(idx).zfill(6) + '.png', image)
    idx = idx+1
    
    # cv2.imshow('MediaPipe Pose', image)
    # if cv2.waitKey(5) & 0xFF == 27:
      # break
cap.release()
'''
'''
for preset in os.listdir('data/vid4/preset_frames/'):
    if preset == ".DS_Store":
        print(preset)
    else:
        name = preset.split("frames_")[1]
        write_to = 'data/vid4/preset_videos/'+name+'.mp4'
        writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=4)
        a = os.listdir(os.path.join('data/vid4/preset_frames/',preset))
        if ".DS_Store" in a:
            a.remove(".DS_Store")

        a.sort()
        for i in a:
            img = Image.open('data/vid4/preset_frames/'+preset+'/'+i)
            writer.append_data(np.asarray(img))
        writer.close()
'''

