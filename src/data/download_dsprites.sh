#! /bin/sh

cd ../data
git clone https://github.com/deepmind/dsprites-dataset.git
cd dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
