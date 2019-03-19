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
    #RATIO = 0.8
    ## Match descriptor and obtain two best matches
    #bf = cv2.BFMatcher()
    #matches = bf.knnMatch(desc_s, desc_t, k=2)

    ## Initialize output variable
    #fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

    #matches_num = len(matches)
    #for i in range(matches_num):
    #    # Obtain the good match if the ration id smaller than 0.8
    #    if matches[i][0].distance <= RATIO * matches[i][1].distance:
    #        temp = np.array([matches[i][0].queryIdx,
    #                         matches[i][0].trainIdx])
    #        # Put points index of good match
    #        fit_pos = np.vstack((fit_pos, temp))
    #return fit_pos

    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(desc_s,desc_t)
    fit_pos = np.array([], dtype=np.int32).reshape((0, 2))
    matches_num = len(matches)
    for i in range(matches_num):
            temp = np.array([matches[i].queryIdx,
                             matches[i].trainIdx])
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
    #fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video = cv2.VideoWriter( video, fourcc, 30.0, (1280,720))
    video = cv2.VideoWriter( video, fourcc, 30.0, (1280,720))
    
    frame = cv2.imread('./logo.jpg')
    frame = cv2.resize(frame, (1280,720))
    print (frame.shape)
    i = 0
    while True:
        video.write(frame)
        i += 1
        if i == 60:
            break

    filelist = os.listdir(path)
    print ('to generate video from ', len(filelist))
    i = -1
    #for item in filelist:
    while True:
            i +=1
            if i == 100000:
                break
            print (i)
            if i<start_frame or i > end_frame:
                continue
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

#step1+2
start_frame = 1700
#end_frame = 2100
end_frame = 1800
##step3
#start_frame = 2400
#end_frame = 2900
#
##step4+5
#start_frame = 3300
#end_frame = 3800
#
##step6
#start_frame = 4300
#end_frame = 4500
#
##step7
#start_frame = 4700
#end_frame = 4920
#
#step8
#start_frame = 5150
#end_frame = 5300
#
##step9-1
#start_frame = 5500
#end_frame = 5750
#
#step9-2
#start_frame = 5800
#end_frame = 5900
#
##step9-3
#start_frame = 5950
#end_frame = 6050
#
##step10
#start_frame = 6200
#end_frame = 6450
#
##step11
#start_frame = 6600
#end_frame = 6800
#
##step12
#start_frame = 7300
#end_frame = 7500
#
##step13
#start_frame = 7800
#end_frame = 8350
#
##step14
#start_frame = 8400
#end_frame = 8850

#test misunderstanding
#start_frame = 300
#end_frame = 400

#demo
#start_frame = 1479
#end_frame = 9020
#start_frame = 7000
#end_frame = 8000

FRAME_DIR = './matched/'

def get_matched_bound(kp_src, bound_point_list):
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

    return bound_list

def remove_duplicates(kp_src, kp_label):
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

    return kp_src, kp_label

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
    time_window_top2 = {}
    for index in range(len(filelist)):
        print (index)
        if index<start_frame:
            continue
        if index>end_frame:
            break
        item = 'frame_{}.jpg'.format(index)
        img = cv2.imread('./frames/'+item)
        original_img = img.copy()
        kp, desc = gen_label_feature(img)   
        
        max_mathed_kp = []
        matched_rank_list = []
        matched_name = None
        max_bound_list = []
        total_start = time.time()
        for name in feature_dict:
            start = time.time()
            print (name)
            #print('feature count to check', desc.shape, feature_dict[name][1].shape)
            fit_pos = match_SIFT(desc, feature_dict[name][1])
            kp_src = kp[:,fit_pos[:,0]]
            kp_label = feature_dict[name][0][:,fit_pos[:,1]]
            end = time.time()
            #print ('match cost: ', end-start)
            start = time.time()

            if len(kp_src[0]) <= 3 or len(kp_label[0]) <=3:
                continue;

            print('point count to check', kp_src.shape, kp_label.shape)
            _, _, inliers, bound_point_list = Ransac(3, 1).ransac_fit(kp_src, kp_label)
            end = time.time()
            #print ('ransac time cost: ', end-start)
            start = time.time()
            if len(inliers[0]) == 0:
                print ('ransac can not find')
                continue
            else:
                print ('ransac pint count is: ', len(inliers[0]))

            bound_list = get_matched_bound(kp_src, bound_point_list)

            kp_src = kp_src[:, inliers[0]]
            kp_label = kp_label[:, inliers[0]]

            kp_src,kp_label = remove_duplicates(kp_src, kp_label)

            print ('after remove duplicate point count is:', len(kp_src[0]))
            if len(kp_src[0]) <= 3:
                continue
            else:
                matched_rank_list.append([name[:-4],len(kp_src[0])])
            if len(max_mathed_kp) == 0 or len(kp_src[0])>len(max_mathed_kp[0]):
                max_mathed_kp = kp_src
                matched_name = name
                max_bound_list = bound_list
            end = time.time()

        matched_rank_list = sorted(matched_rank_list, key=lambda item:item[1], reverse=True)
        all_matched_dict[index] = matched_rank_list

        if len(matched_rank_list) >= 4:
            matched_rank_list = matched_rank_list[:4]
        else:
            matched_rank_list += [['None',0]]*(4-len(matched_rank_list))

        top1 = matched_rank_list[0]
        if '.' in top1[0]:
            top1[0] = top1[0][:top1[0].index('.')]
        top2 = matched_rank_list[1]
        if '.' in top2[0]:
            top2[0] = top2[0][:top2[0].index('.')]
        time_window_top2[index] = [top1,top2]

        draw_info_by_matched(item, index, time_window_top2, original_img, max_mathed_kp, kp, max_bound_list, matched_name)
    f = open('./'+str(start_frame)+'.rank.txt', 'w')
    for item in all_matched_dict:
        value = all_matched_dict[item]
        f.write(str(item)+','+str(value[0][0])+','+str(value[0][1])
                         +','+str(value[1][0])+','+str(value[1][1])
                         +','+str(value[2][0])+','+str(value[2][1])
                         +','+str(value[3][0])+','+str(value[3][1])+'\n')
    f.close()

def draw_info_by_matched(item, index, time_window_top2, original_img, max_mathed_kp, kp, max_bound_list, matched_name):
    distance = 30
    if index-distance in time_window_top2:
        time_window_top2.pop(index-distance)
    print ('after pop : ',time_window_top2)
    label = []
    for k in time_window_top2:
        top1 = time_window_top2[k][0][0]
        if top1 != 'None':
            label.append(top1)
        #top2 = time_window_top2[k][1][0]
        #if top2 != 'None':
        #    label.append(top2)
    if len(label) == 0:
        most_common = 'None'
    else:
        most_common = collections.Counter(label).most_common(1)[0][0]
    #print ('label', label)
    print ('most common:', most_common)
    sum_most_common = 0
    total = 0
    common_count = 0
    for k in time_window_top2:
        total += time_window_top2[k][0][1]
        #total += time_window_top2[k][1][1]
        if time_window_top2[k][0][0] == most_common:
            sum_most_common += time_window_top2[k][0][1]
            common_count += 1
        #if time_window_top2[k][1][0] == most_common:
        #    sum_most_common += time_window_top2[k][1][1]
    print (most_common, 'is ', sum_most_common, total, float(sum_most_common)/(total+1e-6))     
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_img, 'frame '+ str(index), (30,680),font,1, (0,0,255), 3)
    if len(time_window_top2) == distance and float(sum_most_common)/(total+1e-6) > 0.75 and sum_most_common > 50 and common_count > distance*0.5:
        print ('recongize as:', most_common)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_img, most_common,  (1000,680),font,2, (0,0,255), 10)

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
                for b in max_bound_list:
                    cv2.rectangle(original_img,(int(b[0]),int(b[2])), (int(b[1]),int(b[3])), (0,255,0), 3)

    print (FRAME_DIR+item)
    cv2.imwrite(FRAME_DIR+item, original_img)
    total_end = time.time()

def test_by_label(src, label):
    img_src = cv2.imread(src)
    img_src_ori = img_src.copy()
    img_label = cv2.imread(label)
    img_label_ori = img_label.copy()
    print (img_src.shape, img_label.shape)
    kp_src, desc_src = gen_label_feature(img_src)   
    kp_label, desc_label = gen_label_feature(img_label)   
    fit_pos = match_SIFT(desc_src, desc_label)
    print ('fitpos',fit_pos)
    kp_src = kp_src[:,fit_pos[:,0]]
    kp_label = kp_label[:,fit_pos[:,1]]

    src_bound_list =[]
    label_bound_list =[]
    _, _, inliers, bound_point_list = Ransac(3, 1).ransac_fit(kp_src, kp_label)

    for b_p in bound_point_list:
        arr_p = kp_src[:,b_p]
        #print (len(arr_p[0][0]))
        bound = (min(arr_p[0][0]), max(arr_p[0][0]), min(arr_p[1][0]), max(arr_p[1][0]))
        src_bound_list.append(bound)

    for b_p in bound_point_list:
        arr_p = kp_label[:,b_p]
        #print (len(arr_p[0][0]))
        #print ('index', b_p)
        #for x in zip(arr_p[0][0],arr_p[1][0]):
        #    print (x)
        #print ('point',(arr_p[0][0]))
        bound = (min(arr_p[0][0]), max(arr_p[0][0]), min(arr_p[1][0]), max(arr_p[1][0]))
        label_bound_list.append(bound)

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
        cv2.circle(img_src_ori, (int(kp_src[0][i]), int(kp_src[1][i])), 10, (0,0,255), 2)      
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_src_ori, str(i), (int(kp_src[0][i]),int(kp_src[1][i])),font,1, (0,0,255), 2)
    for b in src_bound_list:
        cv2.rectangle(img_src_ori,(int(b[0]),int(b[2])), (int(b[1]),int(b[3])), (0,255,0), 2)
    cv2.imwrite('./test_src.jpg', img_src_ori)

    print (kp_label)
    print (kp_src)
    for i in range(len(kp_label[0])):
        cv2.circle(img_label_ori, (int(kp_label[0][i]), int(kp_label[1][i])), 10, (0,0,255), 2)      
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_label_ori, str(i), (int(kp_label[0][i]),int(kp_label[1][i])),font,1, (0,0,255), 2)
    for b in label_bound_list:
        cv2.rectangle(img_label_ori,(int(b[0]),int(b[2])), (int(b[1]),int(b[3])), (0,255,0), 2)
    cv2.imwrite('./test_label.jpg', img_label_ori)


#split_to_frames('./data/VID_20190225_145649.mp4')
match_by_feature('./frames/', './label/')
gen_video_from_frames(FRAME_DIR, './result/'+str(start_frame)+'_detect.avi')
#test_by_label('./frames/frame_3815.jpg', './label/step_14.jpg')

