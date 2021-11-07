
import os, argparse, time, subprocess, io, shlex
import pandas as pd
import tqdm
import h5py
import numpy as np
from PIL import Image
Image.warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='ImageNet Validation')

parser.add_argument('--in_path', required=False,
                    help='path to ImageNet folder that contains val folder')
parser.add_argument('--batch_size', default=128, type=int,
                    help='size of batch for validation')
parser.add_argument('--workers', default=20,
                    help='number of data loading workers')
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet50', 'resnet50_at', 'cornets'], default='resnet50',
                    help='back-end model architecture to load')

FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


set_gpus(FLAGS.ngpus)

import torch
import torch.nn as nn
import torchvision
from vonenet import get_model

device = torch.device("cuda" if FLAGS.ngpus > 0 else "cpu")


def val():
    model = get_model(model_arch=FLAGS.model_arch, pretrained=True)
    transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5]),
            ])
    if FLAGS.ngpus == 0:
        print('Running on CPU')
    if FLAGS.ngpus > 0 and torch.cuda.device_count() > 1:
        print('Running on multiple GPUs')
        model = model.to(device)
    elif FLAGS.ngpus > 0 and torch.cuda.device_count() is 1:
        print('Running on single GPU')
        model = model.to(device)
    else:
        print('No GPU detected!')
        model = model.module
    model.eval()
    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        _model_feats.append(np.reshape(output, (len(output), -1)))

    try:
        m = model.module
    except:
        m = model
    m=m.model
    # print(m)
    model_layer = getattr(getattr(m, 'V4'), 'conv1')
    model_layer.register_forward_hook(_store_feats)
    f = h5py.File('/content/gdrive/MyDrive/npc_v4_data.h5','r')
    # if FLAGS.session=='natural':
    #   data = f['images/naturalistic'][:]
    # else:
      # session_path=FLAGS.session.replace('_','/')
      # final_path=session_path[:-1]+'_'+session_path[-1:]
      # data = f['images/synthetic/monkey_'+final_path][:]
    data = f['images/synthetic/monkey_m/ohp/session_1'][:]
    with torch.no_grad():
        model_feats = []
        #fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.*')))
        # if len(fnames) == 0:
        #     raise FileNotFoundError(f'No files found in {FLAGS.data_path}')
        for fname in tqdm.tqdm(data):
            try:
                im = Image.fromarray(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0).cuda()  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            model_feats.append(_model_feats[0])
        model_feats = np.concatenate(model_feats)


    # model_feats = []
    # validator = ImageNetVal(model)
    # record = validator()

    # print(record['top1'])
    # print(record['top5'])
    return


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.in_path, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5]),
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(device)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    val()
