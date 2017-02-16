from distutils.core import setup

setup(
    name='tnet',
    version='0.1.0',
    packages=['tnet', 'tnet.nn', 'tnet.nn.rnn', 'tnet.core', 'tnet.cuda', 'tnet.meter', 'tnet.engine', 'tnet.dataset',
              'tnet.dataset.custom_datasets', 'tnet.optimizers', 'examples'],
    url='https://github.com/mhjabreel/tnet',
    license='MIT',
    author='Mohammed Jabreel',
    author_email='mhjabreel@gmail.com',
    description='Torch and torchnet like library for building and training neural networks in Theano'
)
