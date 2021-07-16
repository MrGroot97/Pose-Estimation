import cv2
import mediapipe as mp
import os
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def create_dir(path):
    if not os.path.exists(path):
            os.makedirs(path)

# For static images:
# IMAGE_FILES = ["input_image.jpeg"]
# with mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     min_detection_confidence=0.5) as pose:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     print(
#         f'Nose coordinates: ('
#         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#     )
#     # Draw pose landmarks on the image.
#     annotated_image = image.copy()
#     mp_drawing.draw_landmarks(
#         annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     cv2.imwrite('annotated_image' + str(idx) + '.png', annotated_image)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo script for human segmentation')
    parser.add_argument("--vid",default='video',help = 'Path to input video directory')
    parser.add_argument("--out_dir",default='outputs',help = 'Path to output directory')

    args = parser.parse_args()

    out_dir  = args.out_dir
    create_dir(out_dir)
    video_dir = args.vid
    for vid_path in os.listdir(video_dir):
        if vid_path.endswith(".DS_Store"):
            continue
        print("processing video",vid_path)
        vidObj = cv2.VideoCapture(os.path.join(video_dir,vid_path))
        frame_width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))   
        frame_height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(out_dir,os.path.basename(vid_path).split('.')[0] + '.mp4'),fourcc, 30.0, (frame_width,frame_height))

        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            while vidObj.isOpened():
                success, image = vidObj.read()
                # print("......",vidObj.isOpened())
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = pose.process(image)
                print("results ....",results.pose_landmarks)
                for i in mp_holistic.PoseLandmark:
                    print("mp_holistic.PoseLandmark ... ",str(i))
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                out.write(image)
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            vidObj.release()
            out.release()
            cv2.destroyAllWindows()