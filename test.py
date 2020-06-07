import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision

import PIL.Image as pil
import cv2
import argparse
import os

from networks import encoder as ENC
from networks import decoder as DEC
from networks import layers as LYR
from util.vis import *
from util.warp import *

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=18)
parser.add_argument('--num_input_images', type=int, default=2)
parser.add_argument('--num_output_channels', type=int, default=2)
parser.add_argument('--scales', type=list, default=[0, 1, 2, 3])
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--video_path', type=str, default='raw_dataset/fda66a0473ef4396.mov')
parser.add_argument('--save_gif', type=str, default='./teaser/fda66a0473ef4396.gif')
parser.add_argument('--checkpoint_dir', type=str, default='./snapshots_simple')
parser.add_argument('--model_name', type=str, default='default.pth')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define ssim
ssim = LYR.SSIM().to(device)

# define simple encoder
simple_encoder = ENC.ResnetEncoder(
    num_layers=args.num_layers, 
    pretrained=True, 
    num_input_images=args.num_input_images
).to(device)

# define simple decoder
simple_decoder = DEC.FlowDecoder(
    num_ch_enc=simple_encoder.num_ch_enc, 
    scales=args.scales, 
    num_output_channels=args.num_output_channels, 
    use_skips=True
).to(device)


if not os.path.exists("./teaser"):
    os.mkdir("teaser")

def load_checkpoint():
    global simple_encoder
    global simple_encoder

    # load simple model
    path = os.path.join(args.checkpoint_dir, 'checkpoints', args.model_name)
    state = torch.load(path)
    simple_encoder.load_state_dict(state['encoder'])
    simple_decoder.load_state_dict(state['decoder'])
    simple_encoder.eval()
    simple_decoder.eval()


def procees_frame(frame: np.array):
    frame = frame[...,::-1].astype(np.float)
    if frame.max() > 1:
        frame /= 255.

    frame = torch.tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
    frame = frame.to(device)
    frame = F.interpolate(frame, (args.height, args.width))
    return frame.float()


def pipeline(prev_frame: np.array, frame: np.array):
    prev_frame = procees_frame(prev_frame)
    frame = procees_frame(frame)

    # get output from simple model
    input = torch.cat([prev_frame, frame], dim=1)
    with torch.no_grad():
        enc_output = simple_encoder(input)
        dec_output = simple_decoder(enc_output)
        flow = dec_output[('flow', 0)]

    flow = flow.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    color_simple_flow = flow_to_color(flow)

    return color_simple_flow[..., ::-1]

def test_video():
    all_frames = []
    cap = cv2.VideoCapture(args.video_path)
    ret, prev_frame = cap.read()
    prev_frame = prev_frame[:320, ...]

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[:320]
        flow = pipeline(prev_frame, frame)
        prev_frame = frame

        frame = cv2.resize(frame, (512, 256))
        flow = cv2.resize(flow, (512, 256))
        frame_flow = np.concatenate([frame, flow], axis=0)

        # Display the resulting frame
        all_frames.append(frame_flow)
        cv2.imshow('frame', frame_flow)
        if cv2.waitKey(200) & 0xFF == ord('q'):
           break

    # save gif
    all_frames = [pil.fromarray(frame[...,::-1]) for frame in all_frames]
    all_frames[0].save(
        args.save_gif, 
        save_all=True, 
        append_images=all_frames[1:],
        duration=200
    )

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_checkpoint()
    test_video()
