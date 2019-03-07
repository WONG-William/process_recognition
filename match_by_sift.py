import numpy as np
import os
import cv2
from affine_ransac import Ransac
import collections
import time

def gen_label_feature(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract key points and SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(img_gray, None)

    # Extract positions of key points
    kp = np.array([p.pt for p in kp]).T

    return kp, desc


def match_SIFT(desc_s, desc_t):
    RATIO = 0.8
    # Match descriptor and obtain two best matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_s, desc_t, k=2)

    # Initialize output variable
    fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

    matches_num = len(matches)
    for i in range(matches_num):
        # Obtain the good match if the ration id smaller than 0.8
        if matches[i][0].distance <= RATIO * matches[i][1].distance:
            temp = np.array([matches[i][0].queryIdx,
                             matches[i][0].trainIdx])
            # Put points index of good match
            fit_pos = np.vstack((fit_pos, temp))

    return fit_pos

def split_to_frames(video):
    cap = cv2.VideoCapture(video)
    print (cap.get(cv2.CAP_PROP_FPS))
    i = 0
    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
    #    (h, w) = frame.shape[:2]
     #   center = (w // 2, h // 2)
     #   M = cv2.getRotationMatrix2D(center, 180, 1.0)
     #   frame = cv2.warpAffine(frame, M, (w, h)) 
        cv2.imwrite('./frames/frame_{}.jpg'.format(i),frame)
        i +=1
        print (i)
    #    if i == 10:
    #        break
    cap.release()

def gen_video_from_frames(path, video):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter( video, fourcc, 30.0, (1280,720))

    filelist = os.listdir(path)
    print ('to generate video from ', len(filelist))
    i = -1
    for item in filelist:
            i +=1
            print (i)
            #if i<1800 or i > 2000:
                #continue
            item = 'frame_{}.jpg'.format(i)
            frame = cv2.imread(path+item)
            if frame is None:
                continue

     #       (h, w) = frame.shape[:2]
      #      center = (w // 2, h // 2)
      #      M = cv2.getRotationMatrix2D(center, 180, 1.0)
      #      frame = cv2.warpAffine(frame, M, (w, h)) 
            
            video.write(frame)
    video.release()

def match_by_feature(src_path, label_path):
    feature_dict = {}
    filelist = os.listdir(label_path)
    for f in filelist:
        img = cv2.imread(label_path+f)
        kp, desc = gen_label_feature(img)   
        feature_dict[f] = (kp,desc)
        print (f, kp.shape)

    filelist = os.listdir(src_path)
    all_matched_dict = {}
    bound_dict = {}
    time_window_top2 = {}
    for index in range(len(filelist)):
        print (index)
        #if index<1600:
        #if index<1800:
            #continue
        #if index>3000:
        #if index>2000:
            #break
        item = 'frame_{}.jpg'.format(index)
        img = cv2.imread('./frames/'+item)
        original_img = img.copy()
        kp, desc = gen_label_feature(img)   
        
        max_mathed_kp = []
        matched_rank_list = []
        matched_name = None
        max_bound_list = []
        for name in feature_dict:
            fit_pos = match_SIFT(desc, feature_dict[name][1])
            kp_src = kp[:,fit_pos[:,0]]
            kp_label = feature_dict[name][0][:,fit_pos[:,1]]

            if len(kp_src[0]) <= 3 or len(kp_label[0]) <=3:
                continue;

            _, _, inliers, bound_point_list = Ransac(3, 1).ransac_fit(kp_src, kp_label)
            if len(inliers[0]) == 0:
                continue
            else:
                print ('ransac pint count is: ', len(inliers[0]))

            bound_list = []
            for b_p in bound_point_list:
                arr_p = kp_src[:,b_p]
                bound = (min(arr_p[0][0]), max(arr_p[0][0]), min(arr_p[1][0]), max(arr_p[1][0]))
                bound_list.append(bound)
            bound_list = sorted(bound_list, key=lambda item:(item[1]-item[0])*(item[3]-item[2]), reverse=True)

            temp_bound_list = []
            for small in bound_list:
                b_in = False
                for big in temp_bound_list:
                    if small[0] >= big[0] and small[1] <= big[1] and small[2] >= big[2] and small[3] <= big[3]:
                        b_in = True
                        break
                if b_in is False:
                    temp_bound_list.append(small)
            bound_list = temp_bound_list

            temp_bound_list = []
            for small in bound_list:
                b_merged = False
                for i,big in enumerate(temp_bound_list):
                    inter = (max(small[0], big[0]), min(small[1], big[1]), max(small[2],big[2]), min(small[3], big[3]))
                    area_inter = (inter[1]-inter[0])*(inter[3]-inter[2])
                    small_area = (small[1]-small[0])*(small[3]-small[2])
                    big_area = (big[1]-big[0])*(big[3]-big[2])
                    if small_area * 0.8 < area_inter or big_area * 0.8 < area_inter:
                        b_merged = True
                        temp_bound_list[i] = (min(small[0], big[0]), max(small[1], big[1]), min(small[2],big[2]), max(small[3], big[3]))
                        break
                if b_merged is False:
                    temp_bound_list.append(small)
            bound_list = temp_bound_list

            kp_src = kp_src[:, inliers[0]]
            kp_label = kp_label[:, inliers[0]]

            src_list = []
            label_list = []
            valid = []
            for i in range(len(kp_src[0])):
                p1 =(kp_src[0][i],kp_src[1][i]) 
                p2 = (kp_label[0][i],kp_label[1][i])
                if p1 not in src_list and p2 not in label_list:
                    valid.append(i)
                    src_list.append(p1)
                    label_list.append(p2)
            kp_src = kp_src[:,valid]
            kp_label = kp_label[:,valid]

            print ('after remove duplicate point count is:', len(kp_src[0]))
            if len(kp_src[0]) <= 3:
                continue
            else:
                matched_rank_list.append([name[:-4],len(kp_src[0])])
            if len(max_mathed_kp) == 0 or len(kp_src[0])>len(max_mathed_kp[0]):
                max_mathed_kp = kp_src
                matched_name = name
                max_bound_list = bound_list
        
        matched_rank_list = sorted(matched_rank_list, key=lambda item:item[1], reverse=True)
        if len(matched_rank_list) >= 4:
            matched_rank_list = matched_rank_list[:4]
        else:
            matched_rank_list += [['None',0]]*(4-len(matched_rank_list))
        all_matched_dict[index] = matched_rank_list
        if len(max_mathed_kp) == 0:
            print ('feature point count is: 0')
        else:
            print ('feature point count is: ',len(max_mathed_kp[0]))
            print ('feature name is: ', matched_name[:-4])
            if len(max_mathed_kp[0]) >= 3:
                for i in range(len(max_mathed_kp[0])):
                    cv2.circle(original_img, (int(max_mathed_kp[0][i]), int(max_mathed_kp[1][i])), 10, (0,0,255), 3)      
                for i in range(len(max_mathed_kp[0])):
                    cv2.circle(original_img, (int(max_mathed_kp[0][i]), int(max_mathed_kp[1][i])), 10, (0,0,255), 3)      
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(original_img, matched_name[:-4]+':'+str(float(len(max_mathed_kp[0]))/len(kp[0]))[:5]
                                    +'('+str(len(max_mathed_kp[0]))+')', (30,30),font,1, (0,0,255), 3)
                bound_dict[index] = bound_list
                for b in max_bound_list:
                    cv2.rectangle(original_img,(int(b[0]),int(b[2])), (int(b[1]),int(b[3])), (0,255,0), 3)

        top1 = matched_rank_list[0]
        top2 = matched_rank_list[1]
        time_window_top2[index] = [top1,top2]
        print (time_window_top2)
        distance = 30
        if index-distance in time_window_top2:
            time_window_top2.pop(index-distance)
        print ('after pop : ',time_window_top2)
        label = []
        for k in time_window_top2:
            label.append(time_window_top2[k][0][0])
            label.append(time_window_top2[k][1][0])
        most_common = collections.Counter(label).most_common(1)[0][0]
        #print ('label', label)
        #print ('most common:', most_common)
        sum_most_common = 0
        total = 0
        for k in time_window_top2:
            total += time_window_top2[k][0][1]
            total += time_window_top2[k][1][1]
            if time_window_top2[k][0][0] == most_common:
                sum_most_common += time_window_top2[k][0][1]
            if time_window_top2[k][1][0] == most_common:
                sum_most_common += time_window_top2[k][1][1]
        print (sum_most_common, total, float(sum_most_common)/(total+1e-6))     
        if len(time_window_top2) == distance and most_common != 'None' and float(sum_most_common)/(total+1e-6) > 0.7:
            print ('recongize as:', most_common)
            cv2.putText(original_img, most_common,  (1000,680),font,2, (0,0,255), 10)

        print (original_img.shape)
        print ('./matched/'+item)
        cv2.imwrite('./matched/'+item, original_img)

    f = open('./rank.txt', 'w')
    for item in all_matched_dict:
        value = all_matched_dict[item]
        f.write(str(item)+','+str(value[0][0])+','+str(value[0][1])
                         +','+str(value[1][0])+','+str(value[1][1])
                         +','+str(value[2][0])+','+str(value[2][1])
                         +','+str(value[3][0])+','+str(value[3][1])+'\n')
    f.close()

    f = open('./bound.txt', 'w')
    for item in bound_dict:
        value = all_matched_dict[item]
        f.write(str(item)+','+str(value[0])+','+str(value[1])
                         +','+str(value[2])+','+str(value[3])+'\n')
    f.close()

def test_by_label(src, label):
    img_src = cv2.imread(src)
    img_src_ori = img_src.copy()
    img_label = cv2.imread(label)
    img_label_ori = img_label.copy()
    print (img_src.shape, img_label.shape)
    kp_src, desc_src = gen_label_feature(img_src)   
    kp_label, desc_label = gen_label_feature(img_label)   
    fit_pos = match_SIFT(desc_src, desc_label)
    print (fit_pos)
    kp_src = kp_src[:,fit_pos[:,0]]
    kp_label = kp_label[:,fit_pos[:,1]]

    _, _, inliers = Ransac(3, 1).ransac_fit(kp_src, kp_label)
    kp_src = kp_src[:, inliers[0]]
    kp_label = kp_label[:, inliers[0]]

    src_list = []
    label_list = []
    valid = []
    for i in range(len(kp_src[0])):
        p1 =(kp_src[0][i],kp_src[1][i]) 
        p2 = (kp_label[0][i],kp_label[1][i])
        if p1 not in src_list and p2 not in label_list:
            valid.append(i)
            src_list.append(p1)
            label_list.append(p2)
    kp_src = kp_src[:,valid]
    kp_label = kp_label[:,valid]
            
    for i in range(len(kp_src[0])):
        cv2.circle(img_src_ori, (int(kp_src[0][i]), int(kp_src[1][i])), 10, (0,0,255), 0)      
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_src_ori, str(i), (int(kp_src[0][i]),int(kp_src[1][i])),font,0.5, (0,0,255))
    cv2.imwrite('./test_src.jpg', img_src_ori)

    print (kp_label)
    print (kp_src)
    for i in range(len(kp_label[0])):
        cv2.circle(img_label_ori, (int(kp_label[0][i]), int(kp_label[1][i])), 10, (0,0,255), 0)      
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_label_ori, str(i), (int(kp_label[0][i]),int(kp_label[1][i])),font,0.5, (0,0,255))
    cv2.imwrite('./test_label.jpg', img_label_ori)


split_to_frames('./data/VID_20190225_145649.mp4')
match_by_feature('./frames/', './label/')
gen_video_from_frames('./matched/', './result/detect.avi')
#test_by_label('./frames/frame_23.jpg', './label/step_7.jpg')

