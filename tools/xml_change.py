import os
import xml.etree.ElementTree as ET

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

def change_xml(xml_name,x_bias,y_bias,out_xml):
    tree = ET.parse(xml_name)
    root = tree.getroot()
    # objs = root.findall('object')

    for obj in root.findall('object'):
        cls_name = obj.find('name').text

        xmlbox = obj.find('bndbox')
        xmin = xmlbox.find('xmin').text
        xmax = xmlbox.find('xmax').text
        ymin = xmlbox.find('ymin').text
        ymax = xmlbox.find('ymax').text

        xmin = int(xmin)-x_bias
        xmax = int(xmax)-x_bias
        ymin = int(ymin)-y_bias
        ymax = int(ymax)-y_bias

        xmlbox.find('xmin').text=str(xmin)
        xmlbox.find('xmax').text=str(xmax)
        xmlbox.find('ymin').text=str(ymin)
        xmlbox.find('ymax').text=str(ymax)

    tree.write(out_xml)






videopath_list=(
    "/home/lishuang/Disk/nfs/traindataset/child300/child_714",
    "/home/lishuang/Disk/nfs/traindataset/child300/child_518",
    "/home/lishuang/Disk/nfs/traindataset/child300/child_28m",
    "/home/lishuang/Disk/nfs/traindataset/child300/child_29m",
    "/home/lishuang/Disk/nfs/traindataset/child300/baby",
    "/home/lishuang/Disk/nfs/traindataset/child300/child",
    "/home/lishuang/Disk/nfs/traindataset/double300/double_619",
    "/home/lishuang/Disk/nfs/traindataset/double300/double_tx2",
    "/home/lishuang/Disk/nfs/traindataset/single300/single_619",
    "/home/lishuang/Disk/nfs/traindataset/single300/single_714",
    "/home/lishuang/Disk/nfs/traindataset/single300/single_tx2",
)

xml_path='/home/lishuang/Disk/gitlab/traincode/mega/tools/inference/videotraindataset_xmlnew'
xml_exist_list = os.listdir(xml_path)
for videopath in videopath_list:
    # savepath=videopath+'_frame'
    basedirname=videopath.split('/')[-1]
    # if os.path.exists(savepath) == False:
    #     os.makedirs(savepath)
    if basedirname not in xml_exist_list:
        continue

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

    new_video_name = []
    new_video_name_dic = {}
    new_x_list = []
    new_y_list = []
    new_w_list = []
    new_h_list = []
    new_csv_path = os.path.join(os.path.join(videopath, ".."), f'{basedirname}_video_cut_new.csv')
    with open(new_csv_path) as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.rstrip()
            items = line.split(',')
            new_video_name.append(items[1])
            new_video_name_dic[items[1]] = [items[2], items[3], items[4], items[5]]
            new_x_list.append(items[2])
            new_y_list.append(items[3])
            new_w_list.append(items[4])
            new_h_list.append(items[5])

    video_list = os.listdir(videopath)
    video_cat=os.path.split(videopath)[-1]
    for video_file in video_list:
        # video_path = os.path.join(xml_path, video_file)
        # if not os.path.isfile(video_path):
        #     print('*** is not a file {}'.format(video_path))
        #     continue
        # videobasename = os.path.basename(video_path)
        filebasename, extension = os.path.splitext(video_file)
        x1, y1, x2, y2 = video_name_dic[filebasename]
        new_x1, new_y1, new_x2, new_y2 = new_video_name_dic[filebasename]
        x1, y1, x2, y2=int(x1), int(y1), int(x2), int(y2)
        new_x1, new_y1, new_x2, new_y2=int(new_x1), int(new_y1), int(new_x2), int(new_y2)
        if video_cat=='child_28m' or video_cat=='child_29m':
            frame_width=1920
            frame_height=1080
            resize_size = get_size((frame_width, frame_height))
            ratio_width = float(resize_size[1] / frame_width)
            ratio_height = float(resize_size[0] / frame_height)
            x1=x1*ratio_width
            x2 = x2 * ratio_width
            new_x1 = new_x1 * ratio_width
            new_x2 = new_x2 * ratio_width
            y1 = y1 * ratio_height
            y2 = y2 * ratio_height
            new_y1 = new_y1 * ratio_height
            new_y2 = new_y2 * ratio_height
        xml_file=os.path.join(xml_path, video_cat,filebasename)
        if os.path.exists(xml_file) == False:
            print(f"{xml_file} is no exists")
            continue
        xml_list = os.listdir(xml_file)
        x_bias=new_x1-x1
        y_bias=new_y1-y1
        for xml_filename in xml_list:
            if xml_filename[-4:]=='.xml':
                xml_name=os.path.join(xml_file, xml_filename)
                if os.path.exists(xml_name) == False:
                    print(f"{xml_name} is no exists")
                    continue
                new_xml_file=xml_file #+'_xml'
                if os.path.exists(new_xml_file) == False:
                    os.makedirs(new_xml_file)
                out_xml=os.path.join(new_xml_file, xml_filename)
                change_xml(xml_name, x_bias, y_bias,out_xml)
