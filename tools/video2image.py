import cv2
import os
import numpy as np


def subfiles(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and not entry.is_dir():
            yield entry.name

def loadAllTagFile( DirectoryPath, tag ):# download all files' name
    result = []
    for file in subfiles(DirectoryPath):
    # for file in os.listdir(DirectoryPath):
        file_path = os.path.join(DirectoryPath, file)
        if os.path.splitext(file_path)[1] == tag:
            result.append(file_path)
    return result

#调整图像尺寸
def get_size(image_size,min_size=480,max_size=640):
    w, h = image_size
    size = min_size
    max_size = max_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)

def video2image(videoname,savepath,total_frame_num):
    # videobasename=os.path.basename(videoname)
    # filebasename,extension=os.path.splitext(videobasename)
    # dirname=os.path.join(savepath,filebasename)
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    dirname = savepath

    cap=cv2.VideoCapture(videoname)
    if cap.isOpened():
        rate = cap.get(5)  # 获取帧率
        fraNum = cap.get(7)  # 获取帧数
        frame_width=cap.get(3)
        frame_height=cap.get(4)
        duration = fraNum / rate #秒
        # with open(openvideo_list, 'a') as f:
        #     f.write(videoname+'\n')
    else:
        cap.release()
        print(f"error video:{videoname}\n")
        return total_frame_num
    #缩小过大的视频尺寸
    if frame_width > 640:
        resize_size=get_size((frame_width,frame_height))
        ratio_width=float(resize_size[1]/frame_width)
        ratio_height = float(resize_size[0] / frame_height)
    #train
    sel_fra_list=np.zeros(15)
    begin_fra=int(fraNum/30)
    fra_add = int(fraNum / 15 + 0.5)
    for idx,sel_fra in enumerate(sel_fra_list):
        sel_fra_list[idx]=begin_fra+idx*fra_add

    if sel_fra_list[14]>=fraNum:
        sel_fra_list[14]=fraNum-1
    # sel_fra=int(fraNum/30)
    # fra_add=int(fraNum/15+0.5)
    videobasename=os.path.basename(videoname)
    filebasename,extension=os.path.splitext(videobasename)
    # x, y, w, h = video_name_dic[filebasename]
    x1, y1, x2, y2 = video_name_dic[filebasename]
    if frame_width > 640:
        x1=int(x1)*ratio_width
        x2=int(x2)*ratio_width
        y1=int(y1)*ratio_height
        y2=int(y2)*ratio_height

    rval=cap.isOpened()
    frame_count = 0
    frame_id=0
    file_data = ""
    # frame_id_tmp=0
    while rval:
        # frame_count += 1

        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id_tmp)
        # print("before frame_id=", cap.get(cv2.CAP_PROP_POS_FRAMES))
        rval, frame = cap.read()
        # print("after frame_id=", cap.get(cv2.CAP_PROP_POS_FRAMES))
        # frame_id_tmp=frame_id_tmp+1

        if rval:
            #train
            #15 frame test
            # if frame_id==15:
            #     break
            # if not sel_fra_list[frame_id]==frame_count:
            #     frame_count+=1
            #     continue
            # frame_id+=1

            # imgmask=np.zeros([480,640],dtype=np.uint8) #mask
            # imgmask[125:413,39:545]=255
            # image=cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=imgmask)

            # image=frame[80:300,50:560]  #裁剪
            # image = frame[int(y):(int(y)+int(h)), int(x):(int(x)+int(w))]
            if frame_width > 640:
                frame=cv2.resize(frame,(resize_size[1],resize_size[0]))
            image = frame[int(y1):(int(y2)), int(x1):(int(x2))]
            #
            # image=frame.copy()
            # image[:,:]=255
            # image[125:413,39:545]=frame[125:413,39:545]
            # cv2.imwrite('{}/{}.jpg'.format(dirname,frame_count), image)
            #ori
            cv2.imwrite('{0}/{1:06d}.jpg'.format(dirname, frame_count), image)
            #test txt
            # file_data += videoname.rstrip('.mp4').rstrip('.avi') + ' ' + str(total_frame_num+1)+' '+str(frame_count)+' '+str(int(fraNum))+ '\n'
            videoname_list = videoname.split('/')
            file_data += videoname_list[-2]+'/'+videoname_list[-1].rstrip('.mp4').rstrip('.avi') + ' ' + str(total_frame_num + 1) + ' ' + str(frame_count) + ' ' + str(int(fraNum)) + '\n'

            #train
            # file_data += videoname.rstrip('.mp4') + ' ' + str(1) + ' ' + str(frame_count) + ' ' + str(int(fraNum)) + '\n'

            # if frame_id==15:
            #     break
            # if not sel_fra_list[frame_id]==frame_count:
            #     pass
            # else:
            #     frame_id += 1
            #     file_data += videoname.rstrip('.mp4') + ' ' + str(1) + ' ' + str(frame_count) + ' ' + str(int(fraNum)) + '\n'


            frame_count += 1
            total_frame_num += 1
    if (fraNum-1) !=frame_count:
        file_data = ""
        total_frame_num=total_frame_num-frame_count
        for vframe_id in range(frame_count):
            file_data += videoname_list[-2] + '/' + videoname_list[-1].rstrip('.mp4').rstrip('.avi') + ' ' + str(total_frame_num + 1) + ' ' + str(vframe_id) + ' ' + str(int(frame_count)) + '\n'
            total_frame_num += 1

            # break
    with open(mega_result, 'a') as f:
        f.write(file_data)


    cap.release()
    return total_frame_num


# videopath="/home/lishuang/Disk/shengshi_data/video_test_split_all/single_897_yinchuan"
# videopath_list=(
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/2020double_company',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/2020double_company_1',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/child_79_company',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/child_137_huaxia',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/child_137_huaxia_1',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/double_54_zhuhai',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/double_54_zhuhai_1',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/double_59_huaxiaxueyuan',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/double_59_huaxiaxueyuan_1',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/double_990_close_company',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/double_beijing',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/double_beijing_1',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/single_28_huaxia',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/single_28_huaxia_2',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/single_897_yinchuan',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/single_897_yinchuan_2',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/single_1000_beijng_shoudu',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/single_1000_guangzhjou',
    # '/home/lishuang/Disk/shengshi_data/video_test_split_all/single_1000_wuhan',
    #             )
videopath_list=(
    # "/home/lishuang/Disk/nfs/traindataset/child300/child_714",
    # "/home/lishuang/Disk/nfs/traindataset/child300/child_518",
    # "/home/lishuang/Disk/nfs/traindataset/child300/child_28m",
    # "/home/lishuang/Disk/nfs/traindataset/child300/child_29m",
    "/home/lishuang/Disk/nfs/traindataset/child300/baby",
    # "/home/lishuang/Disk/nfs/traindataset/child300/child",
    # "/home/lishuang/Disk/nfs/traindataset/double300/double_619",
    # "/home/lishuang/Disk/nfs/traindataset/double300/double_tx2",
    # "/home/lishuang/Disk/nfs/traindataset/single300/single_619",
    # "/home/lishuang/Disk/nfs/traindataset/single300/single_714",
    # "/home/lishuang/Disk/nfs/traindataset/single300/single_tx2"
)
for videopath in videopath_list:
    savepath=videopath+'_frame'
    basedirname=videopath.split('/')[-1]
    if os.path.exists(savepath) == False:
        os.makedirs(savepath)

    video_name = []
    video_name_dic={}
    x_list=[]
    y_list=[]
    w_list=[]
    h_list = []
    csv_path = os.path.join(os.path.join(videopath, ".."), f'{basedirname}_video_cut.csv')
    with open(csv_path) as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.rstrip()
            items = line.split(',')
            video_name.append(items[1])
            video_name_dic[items[1]]=[items[2],items[3],items[4],items[5]]
            x_list.append(items[2])
            y_list.append(items[3])
            w_list.append(items[4])
            h_list.append(items[5])

    # mega_result=os.path.join(savepath,f'{basedirname}.txt')  #存放在生成文件夹内
    mega_result=os.path.join(os.path.join(videopath, ".."),f'{basedirname}.txt')  #存放在文件夹上一级目录
    # openvideo_list=os.path.join(savepath,'video_list.txt')
    file_data = ""
    with open(mega_result, 'w') as f:
        f.write(file_data)
    # with open(openvideo_list, 'w') as f:
    #     f.write(file_data)

    total_frame_num=0
    video_list = os.listdir(videopath)
    vid_num=0
    vid_total=len(video_list)
    #文件夹下视频内容
    for video_file in video_list:
        video_path = os.path.join(videopath, video_file)
        if not os.path.isfile(video_path):
            print('*** is not a file {}'.format(video_path))
            continue
        name, ext = os.path.splitext(video_file)
        dst_directory_path = os.path.join(savepath, name)
        if not os.path.exists(dst_directory_path):
            os.mkdir(dst_directory_path)

        print('[{}/{}]:{} start.\n'.format(vid_num, vid_total, video_file))
        total_frame_num = video2image(video_path, dst_directory_path, total_frame_num)
        print('[{}/{}]:{} has finished.\n'.format(vid_num, vid_total, video_file))
        vid_num = vid_num + 1




#文件夹下多个文件夹
# for video_file in video_list:
#     video_path = os.path.join(videopath, video_file)
#     if not os.path.isdir(video_path):
#         print('*** is not a dir {}'.format(video_path))
#         continue
#     dst_video_path = os.path.join(savepath, video_file)
#     if not os.path.exists(dst_video_path):
#         os.mkdir(dst_video_path)
#
#     vid_list = os.listdir(video_path)
#     vid_total=len(vid_list)
#     vid_num=0
#     for vid_name in vid_list:
#         video_file_path = os.path.join(video_path, vid_name)
#         name, ext = os.path.splitext(vid_name)
#         dst_directory_path = os.path.join(dst_video_path, name)
#         if not os.path.exists(dst_directory_path):
#             os.mkdir(dst_directory_path)
#         print('[{}/{}]:{} start.\n'.format(vid_num, vid_total, vid_name))
#         total_frame_num=video2image(video_file_path, dst_directory_path,total_frame_num)
#
#         print('[{}/{}]:{} has finished.\n'.format(vid_num,vid_total,vid_name))
#         vid_num = vid_num + 1




