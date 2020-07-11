from setuptools import setup

import shutil
from pathlib import Path

# Remove stale transformers.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "yuccnlptools.egg-info"
if stale_egg_info.exists():
    shutil.rmtree(stale_egg_info)

stale_dist = Path(__file__).parent / "dist"
if stale_dist.exists():
    shutil.rmtree(stale_dist)



setup(name='yuccnlptools',
        version='0.0.1',
        description='基于pytorch & transformers的文本工具，用于分类、生成等的训练，用于线上服务的部署',
        long_description=open('README.md', 'r', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/yucc2018/yuccnlptools',
        author='Chen-Chen Yu',
        author_email='6506666@gmail.com',
        license='MIT',
        packages=['yuccnlptools'],
        zip_safe=False)

