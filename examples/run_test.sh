
mkdir -p data

if [ ! -f /data/mnist.pkl.gz ]; then
    echo "Downloading mnist"
    cd data
    wget "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
    cd ..
fi
python test.py