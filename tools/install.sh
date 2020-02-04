# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali-tf-plugin
# git clone https://github.com/NVIDIA/apex.git
# cd apex
# python setup.py install --cpp_ext --cuda_ext
# python setup.py develop
pip install pip --upgrade
sudo apt-get install graphviz
pip install graphviz
pip install --upgrade git+https://github.com/Tramac/torchscope.git
# pip install torch==1.2.0
pip install tensorboard==1.13.0
pip install tensorboardX==1.6
#git clone https://github.com/NVIDIA/apex.git
#cd apex
#python setup.py install --cpp_ext --cuda_ext
#install
wget https://azcopyvnext.azureedge.net/release20190703/azcopy_linux_amd64_10.2.1.tar.gz
tar -xvf azcopy_linux_amd64_10.2.1.tar.gz
rm azcopy_linux_amd64_10.2.1.tar.gz

# download
./azcopy_linux_amd64_10.2.1/azcopy copy https://test26183922662.blob.core.windows.net/v-hongyy/cifar.tar .
tar -xvf cifar.tar
rm cifar.tar

./azcopy_linux_amd64_10.2.1/azcopy copy https://test26183922662.blob.core.windows.net/v-hongyy/imagenet.tar .
tar -xvf imagenet.tar
rm imagenet.tar