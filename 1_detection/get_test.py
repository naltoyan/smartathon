import pandas as pd

import csv
import shutil
from pathlib import Path
import fire



def read_coordinate_file(f: str) -> pd.DataFrame:
    return pd.read_csv(f).reset_index(drop=True)


def get_images(path, output):
    path = Path(path)
    df = read_coordinate_file(path / 'test.csv')
    df['image_path'] = df.image_path.apply(lambda p: path / 'images' / p)
    out = Path(output)
    out_test = out / 'test'
    if not out.exists():
        out.mkdir()
        print('creating dataset folders')
        out_test.mkdir()
        (out_test / 'images').mkdir()
    for img_path in df.image_path:
        new_img = out_test / 'images' / img_path.name
        shutil.copyfile(src=img_path, dst=new_img)


if __name__ == '__main__':
    fire.Fire()

