import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils.crf import dense_crf

from os import listdir
from os.path import join

dir_img = 'test/imgs/'  #test image data folder 
dir_mask = 'test/masks/'  #test image mask data folder
dir_result = 'result/'  #test image data folder

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                use_dense_crf=False):
    net.eval()

    ds = BasicDataset('.', '.', scale=scale_factor)
    img = ds.preprocess(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(1280),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    dice_avg = 0

    net = UNet(n_channels=3, n_classes=1)                                                                          

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    
    image_filenames = [join(dir_img, x) for x in listdir(dir_img)] #read test image data folder
    mask_filenames = [join(dir_mask, x) for x in listdir(dir_mask)] #read test image mask data folder

    out_files = []
    if not args.output:
        for f in image_filenames:
            pathsplit = os.path.splitext(f)
            line = pathsplit[0].replace(dir_img, dir_result) #change location
            out_files.append("{}_OUT{}".format(line, pathsplit[1])) #output file location and name
    elif len(image_filenames) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    for i,fn in enumerate(image_filenames):
        
        img = Image.open(fn)
        ground_truth = Image.open(mask_filenames[i])
        
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=False,
                           device=device)
        
        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
            
        [x_size, y_size] = np.shape(mask)
        mask_area = 0
        groundtruth_area = 0
        Intersection = 0
        dice = 0
        result = np.asarray(result)
        ground_truth = np.asarray(ground_truth)
        for i in range(x_size):
            for j in range(y_size):
                
                if result[i][j] == 255:
                    mask_area = mask_area+1
                    
                if ground_truth[i][j] == 1:
                    groundtruth_area = groundtruth_area+1
                    if (result[i][j] == 255) & (ground_truth[i][j] == 1):
                        Intersection = Intersection+1
        dice = 2*Intersection/(mask_area+groundtruth_area)
        logging.info("dice coefficient:{}".format(dice))
        print ("dice coefficient:",dice)
        dice_avg = dice + dice_avg
        
    dice_avg = dice_avg/len(image_filenames)
    logging.info("average dice coefficient:{}".format(dice_avg))
    print ("average dice coefficient:",dice_avg)

