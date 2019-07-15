# Image Classification Out of the Box

A way to use image classification **without** any programming or machine learning knowledge/experience. 
   
With deep learning knowledge, you can easily create custom convolutional nets by changing the Models/custom.txt file to experiment with different architectures.
  
## Setup:

**Requirements**: Latest version of PyTorch

**Data**: Root Data folder (which should be passed into the program as a command-line argument) should have subdirectories corresponding to the classes the network should predict.  
Should have at least 10 examples per class (if using pre-trained model).

**Example One:**  
── RootDataFolder   
&nbsp;&nbsp;&nbsp;&nbsp;└── Class1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage4  
&nbsp;&nbsp;&nbsp;&nbsp;└── Class2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage4  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage5  
&nbsp;&nbsp;&nbsp;&nbsp;└── Class3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage3  
&nbsp;&nbsp;&nbsp;&nbsp;└── Class4  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── ClassImage3  

**Example Two:**  
── Data   
&nbsp;&nbsp;&nbsp;&nbsp;└── Cats  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── cats1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── cats2.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── cats3.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;└── Dogs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── dogs1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── dogs2.jpg    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── dogs3.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── dogs4.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── dogs4.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;└── Sheep  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── sheep1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └──  sheep2.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └──  sheep3.jpg  



## Network

** Custom **
Create custom convolutional nets by editing the Models/custom.txt file.
  
**Layers implemented so far:**  
  
**conv2d**  
input_channel output_channel kernel_size, stride, padding  
  
**linear**  
input_size output_size  
For linear layers, use IN to automatically calculate input size from conv -> linear. Use OUT as final output size for final layer. 
  
**Pretrained**  
Model is a pretrained VGG-16 network that was trained on the ImageNet dataset.  
The classification layer is dynamic enough to allow for an arbitrary amount of classes. 
  
[ImageNet Link](http://www.image-net.org/ "ImageNet Link")  
  
## CLI

Command-line interface.

**Arguments:**  
Root Data Folder: --dataFolder Folder that holds the data where subfolders are classes  
Epochs: --epochs Number of epochs
Custom: --custom Use of custom nets (1 for yes, 0 for no)
