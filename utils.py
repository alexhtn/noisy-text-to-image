import os
from datetime import datetime

import torch


def get_save_directory():
    return os.path.join('output')


def save_checkpoint(epoch, model, optimizer, prefix='', **kwargs):
    if prefix != '' and not prefix.endswith('-'):
        prefix += '-'

    save_directory = get_save_directory()
    os.makedirs(save_directory, exist_ok=True)

    checkpoint_data = {'epoch': epoch,
                       'model': model.state_dict(),
                       'optimizer': optimizer.state_dict()}
    checkpoint_data.update(kwargs)

    torch.save(checkpoint_data, os.path.join(save_directory, f'{prefix}checkpoint.pth'))


def save_best(model, prefix='', **kwargs):
    if prefix != '' and not prefix.endswith('-'):
        prefix += '-'

    save_directory = get_save_directory()
    os.makedirs(save_directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_directory, f'{prefix}best.pth'))

    txt_path = os.path.join(save_directory, f'{prefix}best.txt')
    if len(kwargs) > 0:
        with open(txt_path, 'w') as f:
            for k, v in kwargs.items():
                f.write(f'{k}: {v}\n')
    else:
        if os.path.exists(txt_path):
            os.remove(txt_path)


def print_and_log(text):
    save_directory = get_save_directory()
    os.makedirs(save_directory, exist_ok=True)

    date_string = datetime.now().strftime('%Y-%m-%d %H:%M')
    text = f'[{date_string}] {text}'

    print(text)

    log_path = os.path.join(save_directory, 'log.txt')

    with open(log_path, 'a') as f:
        f.write(text + '\n')
