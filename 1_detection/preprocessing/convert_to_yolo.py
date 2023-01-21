import fire
import pandas as pd
import numpy as np
import imagesize

import shutil
from pathlib import Path


def read_coordinate_file(f: str) -> pd.DataFrame:
    return pd.read_csv(f).reset_index(drop=True)


def write_image(path, info, output):
    path = Path(path)
    new_img = output / 'images' / path.name
    label = output / 'labels' / path.name
    shutil.copyfile(src=path, dst=new_img)
    info[['class', 'x', 'y', 'w', 'h']].to_csv(label.with_suffix('.txt'), sep=' ', index=False, header=None,
                                               float_format='%g')


def create_yaml(df, dirname, filename='smartathon.yaml'):
    return {
        'path': dirname,
        'train': dirname / 'train',
        'val': dirname / 'val',

    }


def create_dataset(base_dir="dataset", output_dir="", val_frac=0.1, seed=700):
    base = Path(base_dir)
    out = Path(output_dir)

    df = read_coordinate_file(base / 'train.csv')

    # extract image size WxH
    df['image_path'] = df.image_path.apply(lambda p: base / 'images' / p)
    img_wh = df.apply(lambda r: imagesize.get(r['image_path']), axis=1)
    df[['img_w', 'img_h']] = pd.DataFrame(img_wh.tolist(), index=df.index)

    # convert to yolo format [x , y , width, hight]
    df['w'] = (df['xmax'] - df['xmin'])
    df['h'] = (df['ymax'] - df['ymin'])
    df['x'] = (df['xmax'] + df['xmin']) / 2
    df['y'] = (df['ymax'] + df['ymin']) / 2

    # fix the coordinates problem by moving the bbx 
    for attr in ['x', 'y']:
        gt = df[attr] > 0
        le = df[attr] < 0
        df.loc[gt, attr] = df[gt][attr] * 2
        df.loc[le, attr] = 0

    # normalize coordinates for yolo format
    df['w'] = df.w / df.img_w
    df['h'] = df.h / df.img_h
    df['x'] = df.x / df.img_w
    df['y'] = df.y / df.img_h

    # tidy up
    df['class'] = df['class'].astype(int)

    out_train = out / 'train'
    out_val = out / 'val'

    if not out.exists():
        print('creating dataset folders')
        out.mkdir()
        out_train.mkdir()
        (out_train / 'images').mkdir()
        (out_train / 'labels').mkdir()
        out_val.mkdir()
        (out_val / 'images').mkdir()
        (out_val / 'labels').mkdir()

    img_paths = df.image_path.drop_duplicates()
    val_imgs = img_paths.sample(frac=val_frac, random_state=seed)
    train_imgs = img_paths.drop(val_imgs.index)

    imgs = df.groupby('image_path')

    for img_path in val_imgs:
        img = imgs.get_group(img_path)
        write_image(img_path, img, out_val)

    for img_path in train_imgs:
        img = imgs.get_group(img_path)
        write_image(img_path, img, out_train)


if __name__ == '__main__':
    fire.Fire()
    # python preprocessing/convert_to_yolo.py create_dataset ./dataset ./yolo_dataset  0.1 500
