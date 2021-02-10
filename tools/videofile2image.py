import os
import numpy as np

def subfiles(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and not entry.is_dir():
            yield entry.name

def loadAllTagFile( DirectoryPath, tag):# download all files' name
    result = []
    for file in subfiles(DirectoryPath):
    # for file in os.listdir(DirectoryPath):
        file_path = os.path.join(DirectoryPath, file)
        if os.path.splitext(file_path)[1] == tag:
            result.append(file_path)
    return result

videopath_list=(
    '/home/lishuang/Disk/nfs/testvideo/baby-tang_rui',
    '/home/lishuang/Disk/nfs/testvideo/child_518_rui_zhang',
    '/home/lishuang/Disk/nfs/testvideo/child_rui_tang',
    '/home/lishuang/Disk/nfs/testvideo/double_619_zhang',
    '/home/lishuang/Disk/nfs/testvideo/single_619_zhang_tang'
)
for videopath in videopath_list:
    vid_list = os.listdir(videopath)
    basedirname = videopath.split('/')[-1]
    mega_result = os.path.join(os.path.join(videopath, ".."), f'{basedirname}.txt')
    for vidname in vid_list:
        vidsinglepath=os.path.join(videopath,vidname)
        fig_names=loadAllTagFile(vidsinglepath,'.jpg')
        fraNum=len(fig_names)

        file_data = ""
        sel_fra_list = np.zeros(15)
        begin_fra = int(fraNum / 30)
        fra_add = int(fraNum / 15 + 0.5)
        for idx, sel_fra in enumerate(sel_fra_list):
            sel_fra_list[idx] = begin_fra + idx * fra_add

        if sel_fra_list[14] >= fraNum:
            sel_fra_list[14] = fraNum - 1

        for sel_fra in sel_fra_list:
            file_data += basedirname+'/'+vidname + ' ' + str(1) + ' ' + str(int(sel_fra)) + ' ' + str(
                int(fraNum)) + '\n'

        with open(mega_result, 'a') as f:
            f.write(file_data)

