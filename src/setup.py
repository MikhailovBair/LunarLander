import io
import re

from setuptools import setup, find_packages


def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# вычищаем локальные версии из файла requirements (согласно PEP440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    # metadata
    name='LunarLander',
    version="0.1",
    license='MIT',
    author='The Great Five',
    author_email="bair1209@gmail.com",
    description='Lunar Lander rl training, python package',
    url='https://github.com/MikhailovBair/LunarLander',

    # options
    packages=find_packages(),
    install_requires=requirements,
)