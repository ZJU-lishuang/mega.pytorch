# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import cv2
from xml.etree import ElementTree as ET

import torch
from tqdm import tqdm

from mega_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, bbox_aug, method, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                if method in ("base", ):
                    images = images.to(device)
                elif method in ("rdn", "mega", "fgfa", "dff"):
                    images["cur"] = images["cur"].to(device)
                    for key in ("ref", "ref_l", "ref_m", "ref_g"):
                        if key in images.keys():
                            images[key] = [img.to(device) for img in images[key]]
                else:
                    raise ValueError("method {} not supported yet.".format(method))
                output = model(images)
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("mega_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

#创建一级分支object
def create_object(root,label,xi,yi,xa,ya):#参数依次，树根，xmin，ymin，xmax，ymax
    #创建一级分支object
    _object=ET.SubElement(root,'object')
    #创建二级分支
    name=ET.SubElement(_object,'name')
    # name.text='AreaMissing'
    name.text = str(label)
    pose=ET.SubElement(_object,'pose')
    pose.text='Unspecified'
    truncated=ET.SubElement(_object,'truncated')
    truncated.text='0'
    difficult=ET.SubElement(_object,'difficult')
    difficult.text='0'
    #创建bndbox
    bndbox=ET.SubElement(_object,'bndbox')
    xmin=ET.SubElement(bndbox,'xmin')
    xmin.text='%s'%xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s'%yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s'%xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s'%ya

#创建xml文件
def create_tree(image_name):
    global annotation
    # 创建树根annotation
    annotation = ET.Element('annotation')
    #创建一级分支folder
    folder = ET.SubElement(annotation,'folder')
    #添加folder标签内容
    folder.text=('ls')

    #创建一级分支filename
    filename=ET.SubElement(annotation,'filename')
    filename.text=image_name.strip('.jpg')

    #创建一级分支path
    path=ET.SubElement(annotation,'path')
    path.text=os.getcwd()+'%s'%image_name.lstrip('.')#用于返回当前工作目录

    #创建一级分支source
    source=ET.SubElement(annotation,'source')
    #创建source下的二级分支database
    database=ET.SubElement(source,'database')
    database.text='Unknown'

    imgtmp = cv2.imread(image_name)
    imgheight,imgwidth,imgdepth=imgtmp.shape
    #创建一级分支size
    size=ET.SubElement(annotation,'size')
    #创建size下的二级分支图像的宽、高及depth
    width=ET.SubElement(size,'width')
    width.text=str(imgwidth)
    height=ET.SubElement(size,'height')
    height.text=str(imgheight)
    depth = ET.SubElement(size,'depth')
    depth.text = str(imgdepth)

    #创建一级分支segmented
    segmented = ET.SubElement(annotation,'segmented')
    segmented.text = '0'

def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        motion_specific=False,
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mega_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, cfg.MODEL.VID.METHOD, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    ##########################################################################
    ## save result img as a video
    # out_video='result.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fgs=25
    # size = (560, 220)
    # video_writer = cv2.VideoWriter(out_video, fourcc, fgs, size)
    outputset=output_folder.split('/')[-1]
    if hasattr(predictions, "copy"):
        score_thr=[0.5,0.7,0.9]
        for _score in score_thr:
            file_data = ""
            with open(f'{outputset}_video_result_{_score}.txt', 'w') as f:
                f.write(file_data)
            predictionstmp = predictions.copy()
            video_list={}
            alarmvideo_list={}
            frame_record=[0,0,0,0,0,0,0,0,0,0]
            frame_num=0
            for image_id, prediction in enumerate(predictionstmp):
                img, target, filename = dataset.get_visualization(image_id)
                frame_id = int(filename.split("/")[-1])
                ## every 12 frames
                # if frame_id%12 != 0:
                #     continue
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_info = dataset.get_img_info(image_id)
                image_width = img_info["width"]
                image_height = img_info["height"]
                prediction = prediction.resize((image_width, image_height))
                pred_bbox = prediction.bbox.numpy()
                pred_label = prediction.get_field("labels").numpy()
                pred_score = prediction.get_field("scores").numpy()
                # keep = pred_score.gt(score_thr)
                keep = pred_score > _score
                boxes = pred_bbox[keep]
                classes = pred_label[keep]
                scores = pred_score[keep]
                template = "{}: {:.2f}"
                boxnum=0
                for box, score,label in zip(boxes, scores,classes):
                    if label ==1:
                        label = 'person'
                        boxnum+=1
                    elif label==2:
                        label = 'head'
                    else:
                        label = 'unknown'
                    rect = [int(k) for k in box[:4]]
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                    s = template.format(label, score)
                    cv2.putText(
                        img, s, (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                    )

                video_dictname=filename.split('/')[-3]+'/'+filename.split('/')[-2]
                if video_dictname not in video_list:
                    frame_record = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    frame_num = 0
                    video_result={}
                    video_list[video_dictname] = video_result
                    alarmvideo_list[video_dictname] = 0
                    file_data_tmp=""
                    for single_video in alarmvideo_list:
                        file_data_tmp += str(single_video) + ', value: ' + str(alarmvideo_list[single_video]) + '\n'
                    with open(f'{outputset}_video_result_{_score}_tmp.txt', 'w') as f:
                        f.write(file_data_tmp)
                
                #every 12 frames
                # if alarmvideo_list[video_dictname] ==0 and boxnum>1:
                #     alarmvideo_list[video_dictname] =1
                #every frame 8 warning
                if boxnum>1:
                    frame_record[frame_num % 10] = 1
                else:
                    frame_record[frame_num % 10] = 0

                if alarmvideo_list[video_dictname] ==0 and sum(frame_record) >7:
                    alarmvideo_list[video_dictname] =1

                frame_num+=1
                if filename.split('/')[-1] not in video_result:
                    video_result[filename.split('/')[-1]]=boxnum

                imgdir=os.path.join(output_folder , os.path.dirname(filename))
                if not os.path.exists(imgdir):
                    os.makedirs(imgdir)
                output_name=os.path.join(imgdir, '{}_{}.jpg'.format(_score,os.path.basename(filename)))
                cv2.imwrite(output_name, img)
                # video_writer.write(img)
                #xml
                create_tree(output_name)
                for box, score,label in zip(boxes, scores,classes):
                    rect = [int(k) for k in box[:4]]
                    create_object(annotation,label, rect[0], rect[1], rect[2], rect[3])

                # for coordinate_list in coordinates_list:
                #     create_object(annotation, coordinate_list[0], coordinate_list[1], coordinate_list[2], coordinate_list[3])
                # if coordinates_list==[]:
                #     break
                # 将树模型写入xml文件
                tree = ET.ElementTree(annotation)
                # tree.write('%s.xml' % output_name.rstrip('.jpg'))

                # tree = ET.ElementTree.parse('%s.xml' % output_name.rstrip('.jpg'))  # 解析movies.xml这个文件
                root = tree.getroot()  # 得到根元素，Element类
                pretty_xml(root, '\t', '\n')  # 执行美化方法
                tree.write('%s.xml' % output_name.rstrip('.jpg'))
            # file_data=str(alarmvideo_list)
            for single_video in alarmvideo_list:
                file_data += str(single_video)+', value: '+str(alarmvideo_list[single_video])+'\n'
            with open(f'{outputset}_video_result_{_score}.txt', 'a') as f:
                f.write(file_data)
    # video_writer.release()

    ##########################################################################
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def inference_no_model(
        data_loader,
        iou_types=("bbox",),
        motion_specific=False,
        box_only=False,
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    dataset = data_loader.dataset

    predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
    print("prediction loaded.")

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
