import matplotlib

matplotlib.use('Agg')
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from time import gmtime, strftime

import os
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# demo.py --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./source.jpg --driving_video ./driving.mp4
# python demo.py --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./1280.jpg --driving_video ./driving.mp4

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

def relative_kp(kp_source, kp_driving, kp_driving_initial):
    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume

    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new


def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])

    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])

    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()

    return inpainting, kp_detector, dense_motion_network


def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network,
                   device, mode='relative'):
    assert mode in ['standard', 'relative']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode == 'relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                      kp_driving_initial=kp_driving_initial)

            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param=None,
                                                dropout_flag=False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def make_animation2(source_image, driving_image_initial, driving_image_final,inpainting_network, kp_detector, dense_motion_network,
                   device, mode='relative'):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving_initial = torch.tensor(driving_image_initial[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving_final = torch.tensor(driving_image_final[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving_initial = driving_initial.to(device)
        driving_final = driving_final.to(device)

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving_initial)
        kp_driving = kp_detector(driving_final)
        kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                              kp_driving_initial=kp_driving_initial)

        dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param=None,
                                                dropout_flag=False)
        out = inpainting_network(source, dense_motion)
        predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


if __name__ == "__main__":

    config = "./config/vox-256.yaml"
    ctpye= '10'
    ctpye1 = '10'
    checkpoints = "./checkpoints/"+ctpye+"/00000099-checkpoint.pth.tar"
    ctpye += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    #checkpoints = "./checkpoints/00000099-checkpoint.pth.tar" # 27:13.71 62:14.15 66:15.01
    # filename = "D:/measure/aoxtest/"
    # drivingpath = './testvideo/driving'
    # drivingrpath = './testvideo/drivingr'
    # x = os.listdir(drivingpath)
    # print(x)
    testset = 'D:/testset/lifeliketest8/'
    #testset = 'D:/testset/testvideor/'
    testnum = os.listdir(testset)
    testdriving = '/driving/'
    testdrivingr = '/drivingr/'
    testvideos = ['driving1.mp4', 'driving2.mp4', 'driving3.mp4', 'driving4.mp4', 'driving5.mp4', 'driving6.mp4',
                  'driving7.mp4']
    print(testnum)
    totaldiff = 0
    totalL1 = 0
    totalfinalL1 = 0
    path = './txt/'+ ctpye +'.txt'
    f = open(path, 'w')
    f.close()
    sourceimg = []
    for num in testnum:
        sourceimg.append(num + '\source.png')
    print(sourceimg)

    for s in sourceimg :
        avgL1 = 0
        totalL1 = 0
        sumd = 0
        for num in testnum:
            sourceimage = testset + s
            # result_video = filename+'/22262/'+drivingvideo
            tmpsource = testset+num+'/tmp'+ctpye1
            re_source = testset + num + '/re'
            fdir = os.makedirs(tmpsource, exist_ok=True)
            fdir = os.makedirs(re_source, exist_ok=True)
            img_shape = (256, 256)
            type = lambda x: list(map(int, x.split(',')))
            mode = 'relative'
            source_image = imageio.imread(sourceimage)
            reader = imageio.get_reader(testset + num + '/' + 'driving.mp4')
            print('ok')
            fps = reader.get_meta_data()['fps']
            driving_video = []
            frame_name = []
            frame_num = -1
            try:
                for im in reader:
                    driving_video.append(im)
                    frame_num += 1
                    frame_name.append(str(frame_num).zfill(7)+'.png')
            except RuntimeError:
                pass
            reader.close()
            if (torch.cuda.is_available() == False):
                device = torch.device('cpu')
            else:
                 device = torch.device('cuda:0')
            source_image = resize(source_image, img_shape)[..., :3]
            driving_video = [resize(frame, img_shape)[..., :3] for frame in driving_video]
            inpainting, kp_detector, dense_motion_network = load_checkpoints(config_path=config,
                                                                                 checkpoint_path=checkpoints,
                                                                                 device=device)

            predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network,
                                             device=device, mode=mode)

            # imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=3)
            for i in range(0,len(frame_name)):
                imageio.imsave(tmpsource+'/'+frame_name[i], img_as_ubyte(predictions[i]))
            for i in range(1, len(frame_name)):
                sourceimager = tmpsource+'/'+ frame_name[i]
                drivingimage_initial = testset+ num + '/driving/' + frame_name[i]
                drivingimage_final = testset + num + '/driving/' + frame_name[0]

                source_imager = imageio.imread(sourceimager)
                driving_image_initial = imageio.imread(drivingimage_initial)
                driving_image_final = imageio.imread(drivingimage_final)

                source_imager = resize(source_imager, img_shape)[..., :3]
                driving_image_initial = resize(driving_image_initial, img_shape)[..., :3]
                driving_image_final = resize(driving_image_final, img_shape)[..., :3]

                predictions = make_animation2(source_imager, driving_image_initial, driving_image_final,inpainting, kp_detector, dense_motion_network,
                                                 device=device, mode=mode)
                imageio.imsave(re_source+'/'+frame_name[i], img_as_ubyte(predictions[-1]))
            sumd = 0
            for i in range(1, len(frame_name)):
                img_gt = cv2.imread(sourceimage)
                img_p = cv2.imread(re_source+'/'+frame_name[i])
                imgdiff = cv2.absdiff(img_gt, img_p)
                diff = np.mean(imgdiff)
                sumd += diff
                print(frame_name[i],diff)
            f = open(path, 'a')
            print(num + " " +s + ':' + str(sumd / (len(frame_name)-1)),len(frame_name))
            f.write(num + " " +s + ':' + str(sumd / (len(frame_name)-1)) + '\n')
            f.close()
            totaldiff += sumd / (len(frame_name)-1)
    f = open(path, 'a')
    f.write('totalfinalL1:' + str(totaldiff / len(sourceimg)**2) + '\n')
    f.close()
    print(totaldiff, len(sourceimg))
    #driving



"""
                sourceimager = tmpsource
                source_imager = imageio.imread(sourceimager)
                reader = imageio.get_reader(testset + num + '/' + testdrivingr + testvideo)
                result = 'testimg/test' + num + '#' + testvideo + '.png'

                fps = reader.get_meta_data()['fps']
                driving_video = []
                try:
                    for im in reader:
                        driving_video.append(im)
                except RuntimeError:
                    pass
                reader.close()

                if (torch.cuda.is_available() == False):
                    device = torch.device('cpu')
                else:
                    device = torch.device('cuda:0')

                source_imager = resize(source_imager, img_shape)[..., :3]
                driving_video = [resize(frame, img_shape)[..., :3] for frame in driving_video]

                predictions = make_animation(source_imager, driving_video, inpainting, kp_detector,
                                             dense_motion_network,
                                             device=device, mode=mode)

                # imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=3)
                imageio.imsave(result, img_as_ubyte(predictions[-1]))
                img_gt = cv2.imread(sourceimage)
                img_p = cv2.imread(result)
                imgdiff = cv2.absdiff(img_gt, img_p)
                diff = np.mean(imgdiff)
                sumd += diff
                print(diff)

            print(num + s, ':', sumd / 7)
            f = open(path, 'a')
            f.write(num + s + ':' + str(sumd / 7) + '\n')
            f.close()
            totaldiff += sumd / 7
        totalL1 = totaldiff / len(testnum)
        totalfinalL1 += totalL1
        print("totaldiff:", totaldiff / len(testnum))
        f = open(path, 'a')
        f.write('totalL1:' + str(totaldiff / len(testnum)) + '\n')
    f.write('totalfinalL1:' + str(totalfinalL1 / len(sourceimg)) + '\n')
    f.close()
    print(totalfinalL1,len(sourceimg)*len(sourceimg))
"""