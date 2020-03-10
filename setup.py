from setuptools import setup, find_packages

setup(
    name='KnowledgeDistillation',
    version="1.0.2",
    description=('A general knowledge distillation framework'),
    long_description=open('README.rst').read(),
    author='ZhangDun',
    author_email='dunnzhang0@gmail.com',
    maintainer='ZhangDun',
    maintainer_email='dunnzhang0@gmail.com',
    license='MIT',
    packages=find_packages(),
    url='https://github.com/DunZhang/KnowledgeDistillation',
    install_requires=['torch>0.4.0'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords="Transformer Networks BERT XLNet PyTorch NLP deep learning"
)
