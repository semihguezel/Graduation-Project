from threading import Thread
import cv2
import time
import numpy as np
#import imutils
from pose_est import *
from unit_sticker import unit_sticker
from e2c import e2c
from cube2sph import map_cube
from rmse import run_rmse
from blue_detection import color_detection

text_calculated_L_BFGS_B_coord = open("txt/coords_calculated_L-BFGS-B_coord.txt", "w", newline='')
text_calculated_BFGS_coord = open("txt/coords_calculated_BFGS_coord.txt", "w", newline='')
text_calculated_L_BFGS_B_rot = open("txt/coords_calculated_L-BFGS-B_rot.txt", "w", newline='')
text_calculated_BFGS_rot = open("txt/coords_calculated_BFGS_rot.txt", "w", newline='')

def w2txt(BFGS, L_BFGS_B):
    np.savetxt(text_calculated_L_BFGS_B_rot, L_BFGS_B[0])
    np.savetxt(text_calculated_BFGS_rot, BFGS[0])
    np.savetxt(text_calculated_L_BFGS_B_coord, L_BFGS_B[1])
    np.savetxt(text_calculated_BFGS_coord, BFGS[1])

def mandel():
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    # sxy = [(3, 1), (0, 1), (1, 1), (2, 1), (1, 0), (1, 2)]
    face_list = ["left", "front", "right", "back", "top", "bottom"]
    scene_change = False
    sticker_select = True
    cw = 256
    k = 8
    # stickers = np.array([[10, 9, 4], [10, 6, 4], [9, 0, 4], [7, 0, 4], [1, 0, 4],[0, 2, 4]],dtype=np.double)
    stickers = np.array([[10, 9, 4], [10, 6, 4], [10, 3, 4], [8, 0, 4], [5, 0, 4], [2, 0, 4], [0, 4, 4] ,[0, 8, 4]],dtype=np.double)
    pixel_stickers = np.zeros((8,3))
    v = cv2.VideoCapture("{your_path}/videos/ublue.mp4")

    while True:
        kk = 0
        ret, frame = v.read()
        a = time.time()
        if ret:
            frame = e2c(frame, sxy=sxy)
        else:
            break

        if sticker_select:
            tracker = cv2.legacy_MultiTracker.create()
            if not scene_change:
                tracker_coordinates = color_detection(frame)
                # print(len(tracker_coordinates))
                for i in range(k):
                    #cv2.imshow("frame", frame)
                    #bbi = cv2.selectROI("frame",frame)
                    tracker_i = cv2.legacy_TrackerMedianFlow.create()
                    tracker.add(tracker_i, frame, (float(tracker_coordinates[i][0]),float(tracker_coordinates[i][1]),float(tracker_coordinates[i][2]),
                    float(tracker_coordinates[i][3])))
                sticker_select = False
                cv2.destroyAllWindows()
            
            else:
                tracker_coordinates = color_detection(frame)
                # print(len(tracker_coordinates))
                for i in range(k):
                    tracker_i = cv2.legacy_TrackerMedianFlow.create()
                    tracker.add(tracker_i, frame, (float(tracker_coordinates[i][0]),float(tracker_coordinates[i][1]),float(tracker_coordinates[i][2]),
                    float(tracker_coordinates[i][3])))
                scene_change = False
                sticker_select = False

        # frame = imutils.resize(frame, width=1920, height=1080)
        (H, W) = frame.shape[:2]
        (success, boxes) = tracker.update(frame)

        frame_copy = frame.copy()
        if success:
            leftmost = min(boxes[:,0])
            rightmost = np.amax(boxes[:,0])
            rightmost_w = boxes[np.nanargmax(boxes[:,0])][2]
            for box in boxes:
                (x,y,w,h) = [int(a) for a in box]
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h),(100,255,0),1)
                # teks = "x:{},y:{}".format(x + int(w / 2), y + int(h / 2))
                # cv2.putText(frame_copy, teks, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # cv2.putText(frame_copy, str(leftmost), (int(boxes[0][0]+10), int(boxes[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame_copy, str(kk+1), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)
                
                if (x+w/2)//cw == 0 and (y+h/2)//cw == 1:
                    face = face_list[0]
                elif (x+w/2)//cw == 1 and (y+h/2)//cw == 1:
                    face = face_list[1]
                elif (x+w/2)//cw == 2 and (y+h/2)//cw == 1:
                    face = face_list[2]
                elif (x+w/2)//cw == 3 and (y+h/2)//cw == 1:
                    face = face_list[3]
                elif (x+w/2)//cw == 1 and (y+h/2)//cw == 0:
                    face = face_list[4]
                elif (x+w/2)//cw == 1 and (y+h/2)//cw == 2:
                    face = face_list[5]

                _u = (x+w/2)%cw
                _v = (y+h/2)%cw

                _u, _v = map_cube(_u,_v,face,cw, 1920, 1080)
                pixel_stickers[kk] = [_u, _v, 1]
                kk = kk + 1

            unit = unit_sticker(pixel_stickers,1920,1080)
            pose = PoseEstimator(stickers, unit)
            pose_solve_L_BFGS_B = pose.solve(method='L-BFGS-B')
            pose_solve_BFGS = pose.solve()
            txt_thread = Thread(target = w2txt, args=(pose_solve_BFGS, pose_solve_L_BFGS_B))
            txt_thread.start()

            # print ('BFGS :', pose.solve())
            # print ('L-BFGS-B:', pose.solve(method='L-BFGS-B'))
        b = time.time()

        cv2.putText(frame_copy, "FPS: {}".format(1/(b-a)), (10, H - ((2 * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # print(1/(b-a))
        cv2.imshow("Frame",frame_copy)

        if int(leftmost) <= 5:
            order = [1,2,3,0,4,5]
            order_face = [3,0,1,2,4,5]

            sxy = [sxy[i] for i in order]
            face_list = [face_list[i] for i in order_face]

            sticker_select = True
            scene_change = True
            # boxes[:,0] = boxes[:,0] + 256

        elif int(rightmost+rightmost_w)  >= 1023:
            order = [3,0,1,2,4,5]
            order_face = [1,2,3,0,4,5]

            sxy = [sxy[i] for i in order]
            face_list = [face_list[i] for i in order_face]

            sticker_select = True
            scene_change = True
            # boxes[:,0] = boxes[:,0] - 256

        if kk == 8:
            sticker_select = True
            scene_change = True

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    v.release()
    cv2.destroyAllWindows()
    text_calculated_L_BFGS_B_coord.close()
    text_calculated_BFGS_coord.close()
    text_calculated_L_BFGS_B_rot.close()
    text_calculated_BFGS_rot.close()
    #run_rmse()

main_thread = Thread(target = mandel) # to run main definiton that algorithm runs
main_thread.start()
