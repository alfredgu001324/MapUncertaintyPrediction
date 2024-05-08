# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from datasets import NuscenesDataset
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)  #8
    parser.add_argument('--pin_memory', type=bool, default=True)  #True
    parser.add_argument('--persistent_workers', type=bool, default=True)   #True
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'train_val', 'val', 'mini_train', 'mini_val'])
    parser.add_argument('--method', type=str, default='base', choices=['base', 'unc'])
    parser.add_argument('--centerline', action='store_true', help='centerline usage')
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True, method=args.method)
    model.method = args.method

    val_dataset = NuscenesDataset(root=args.root, split=args.split, local_radius=model.hparams.local_radius, centerline=args.centerline)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, drop_last=False)
    trainer.validate(model, dataloader)
