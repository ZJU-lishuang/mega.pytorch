import os
import shutil

def subfiles(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and not entry.is_dir():
            yield entry.name

videopath="../../video_mega/"
xmlpath="./inference/video_test"

video_list = os.listdir(videopath)
xml_list = os.listdir(xmlpath)
for class_file in video_list:
    video_path = os.path.join(videopath, class_file)
    xml_path = os.path.join(xmlpath, class_file)
    if not os.path.isdir(video_path):
        print('*** is not a dir {}'.format(video_path))
        continue
    vid_list = os.listdir(video_path)
    for vid_name in vid_list:
        video_file_path = os.path.join(video_path, vid_name)
        xml_file_path = os.path.join(xml_path, vid_name)
        single_video_list=os.listdir(video_file_path)
        result=[]
        for file in subfiles(xml_file_path):
            if os.path.splitext(file)[1] == '.xml':
                result.append(file)
        result.sort()
        single_video_list.sort()
        for idx,sing_video_path in enumerate(single_video_list):
            old_xml_path=os.path.join(xml_file_path,result[idx])
            new_xml_path=os.path.join(video_file_path,sing_video_path.replace('.jpg','.xml'))
            shutil.copy(old_xml_path, new_xml_path)

