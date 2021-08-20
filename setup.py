from setuptools import setup, find_namespace_packages

setup(
    name='vsdkx-addon-group',
    url='https://gitlab.com/natix/cvison/vsdkx/vsdkx-addon-group',
    author='Guja',
    author_email='g.mekokishvili@omedia.ge',
    namespace_packages=['vsdkx', 'vsdkx.addon'],
    packages=find_namespace_packages(include=['vsdkx*']),
    dependency_links=[
        'git+https://gitlab+deploy-token-485942:VJtus51fGR59sMGhxHUF@gitlab.com/natix/cvison/vsdkx/vsdkx-core.git#egg=vsdkx-core',
        'git+https://gitlab+deploy-token-488350:1jt8j5EcWg5gfvRF4Bq1@gitlab.com/natix/cvison/vsdkx/vsdkx-addon-tracking'
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
