from setuptools import setup, find_namespace_packages

setup(
    name='vsdkx-addon-group',
    url='https://gitlab.com/natix/cvison/vsdkx/vsdkx-addon-group',
    author='Guja',
    author_email='g.mekokishvili@omedia.ge',
    namespace_packages=['vsdkx', 'vsdkx.addon'],
    packages=find_namespace_packages(include=['vsdkx*']),
    dependency_links=[
        'https://github.com/natix-io/vsdkx-core.git#egg=vsdkx-core',
        'https://github.com/natix-io/vsdkx-addon-tracking.git#egg=vsdkx-addon-tracking'
    ],
    install_requires=[
        'vsdkx-core',
        'vsdkx-addon-tracking',
        'chinese-whispers==0.7.4',
        'scikit-learn>=0.24.1',
        'networkx==2.5.1',
        'scipy>=1.4.1',
        'numpy>=1.18.5',
    ],
    version='1.0',
)
