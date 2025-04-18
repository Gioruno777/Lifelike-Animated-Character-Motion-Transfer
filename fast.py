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

import os

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


def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


if __name__ == "__main__":


    config = "./config/vox-256.yaml"
    checkpoints = "./checkpoints/10/00000099-checkpoint.pth.tar"
    filename = "D:/measure/voxtest/"
    drivingpath = filename + '/testvideonew'
    x = os.listdir(drivingpath)
    for drivingvideo in x:
        sourceimage = filename+'frame/'+drivingvideo+'/0000000.png'
        result_video1 = filename+'/op1/'+drivingvideo
        img_shape = (256, 256)
        type = lambda x: list(map(int, x.split(',')))
        mode = 'relative'
        find_best_frame = "True"
        source_image = imageio.imread(sourceimage)
        reader = imageio.get_reader(drivingpath+'/'+drivingvideo)
        print('ok')

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

        source_image = resize(source_image, img_shape)[..., :3]
        driving_video = [resize(frame, img_shape)[..., :3] for frame in driving_video]
        inpainting, kp_detector, dense_motion_network = load_checkpoints(config_path=config,
                                                                         checkpoint_path=checkpoints,
                                                                         device=device)

        if find_best_frame == False:
            i = find_best_frame(source_image, driving_video, opt.cpu)
            print("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i + 1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector,
                                                 dense_motion_network, device=device, mode=opt.mode)
            predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector,
                                                  dense_motion_network, device=device, mode=opt.mode)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network,
                                         device=device, mode=mode)

            # imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
            imageio.mimsave(result_video1, [img_as_ubyte(frame) for frame in predictions], fps=30)


