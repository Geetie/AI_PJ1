import os
import shutil
import zipfile

import pandas as pd
import requests
from tqdm.auto import tqdm

from config import CSV_PATH, dataset_path


def download_dataset():
    print(f'数据集目录：{dataset_path}')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    links = pd.read_csv(CSV_PATH)
    for i, link in enumerate(links['link']):
        file_name = links['file'][i]
        file_size_str = links['size'][i] if 'size' in links.columns else '?'
        print(f'Downloading {file_name} ({file_size_str})...')
        file_name = os.path.join(dataset_path, file_name)
        need_download = True
        if os.path.exists(file_name):
            expected_mb = float(file_size_str.replace('MB', '')) * 1024 * 1024 if isinstance(file_size_str, str) and 'MB' in file_size_str else 0
            actual_mb = os.path.getsize(file_name)
            if expected_mb > 0 and abs(actual_mb - expected_mb) / expected_mb > 0.01:
                print(f'  File incomplete ({actual_mb / 1024 / 1024:.1f}MB / {expected_mb / 1024 / 1024:.1f}MB), re-downloading...')
                os.remove(file_name)
            else:
                print(f'  Already exists, skipping.')
                need_download = False
        if need_download:
            if shutil.which('wget'):
                os.system(f'wget -q --show-progress -O "{file_name}" "{link}"')
            elif shutil.which('curl'):
                os.system(f'curl -L -# -o "{file_name}" "{link}"')
            else:
                response = requests.get(link, stream=True)
                total = int(response.headers.get('content-length', 0))
                with open(file_name, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=10485760),
                                      total=total // 10485760 + 1, desc=file_name, unit='MB'):
                        if chunk:
                            f.write(chunk)

    zip_list = ['mchar_train', 'mchar_test_a', 'mchar_val']
    for little_zip in zip_list:
        zip_name = os.path.join(dataset_path, little_zip)
        zip_path = os.path.join(dataset_path, f'{little_zip}.zip')
        need_extract = True
        if os.path.exists(zip_name) and os.path.isdir(zip_name):
            file_count = len(os.listdir(zip_name))
            if file_count > 100:
                print(f'{little_zip}/ already extracted ({file_count} files), skipping.')
                need_extract = False
        if need_extract:
            if os.path.exists(zip_name):
                print(f'{little_zip}/ incomplete, removing and re-extracting...')
                shutil.rmtree(zip_name)
            print(f'Extracting {little_zip}.zip...')
            if shutil.which('unzip'):
                ret = os.system(f'unzip -o "{zip_path}" -d "{dataset_path}"')
                if ret != 0:
                    print(f'  unzip failed (exit code {ret}), trying Python zipfile...')
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(path=dataset_path)
            else:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(path=dataset_path)
            print(f'{little_zip} extraction done.')
