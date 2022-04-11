
import os, argparse, time, subprocess, io, shlex
import pandas as pd
import tqdm
from advertorch.attacks import LinfPGDAttack, L2PGDAttack,L1PGDAttack
import numpy as np

parser = argparse.ArgumentParser(description='ImageNet Validation')

parser.add_argument('--in_path', required=True,
                    help='path to ImageNet folder that contains val folder')
parser.add_argument('--batch_size', default=128, type=int,
                    help='size of batch for validation')
parser.add_argument('--workers', default=20,
                    help='number of data loading workers')
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet50', 'resnet50_at', 'cornets'], default='resnet50',
                    help='back-end model architecture to load')
parser.add_argument('--attack',
                    help='type of attack')

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

    validator = ImageNetVal(model)
    record = validator()

    print(record['top1'])
    print(record['top5'])
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
        min_pixel=0
        max_pixel=0
        with torch.no_grad():
#             adversary = L2PGDAttack(
#             self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=6.0,
#             nb_iter=20, eps_iter=0.75, rand_init=True, clip_min=-1.0, clip_max=1.0,
#             targeted=False)

            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(device)
                inp = inp.to(device)
                output = self.model(inp)
                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)
        accuracy_array=[]
        accuracy_array.append(record['top1'])
        accuracy_array.append(record['top5'])

        np.save(f'/content/gdrive/MyDrive/model_OOD_acc/sketch/VOneCORnet-S_accuracy.npy', accuracy_array)

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
