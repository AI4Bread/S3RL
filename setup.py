from setuptools import setup, find_packages

setup(
    name='S3RL',
    version='0.1.1',
    description='S3RL: Separable Spatial Single-cell Transcriptome Representation Learning via Graph Transformer and Hyperspherical Prototype Clustering',
    author='Penglei Wang',
    author_email='pengleiwangcn@gmail.com',
    packages=find_packages(where='.', include=['S3RL'])
)
