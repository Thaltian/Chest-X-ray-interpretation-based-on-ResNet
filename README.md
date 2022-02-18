# ECE9123-X-ray
This is the term project for ECE9123 by Yang Hu and Hao Tian. 

Since we use visdom to perfrom real-time inspection, please install visdom libraray in python and initate output port(e.g. You can just use "visdom" command in terminal to initate). Of course it is totally fine if you don't initate visdom if you don't want to inspect training process in real time. It won't affect training process of the model.

It is link of our dataset link: https://drive.google.com/file/d/1b1kDyafEz1Mcxozip4miFmeDRsp7hA_W/view?usp=sharing

Please ensure the dataset and the training code is on the same folder. And then you can just run "transfer_train.py" to start. The "transfer_train.py" will also use "utils.py" and "xray_dataset_pretreat.py". So please also ensure "utils.py" and "xray_dataset_pretreat.py" are also in the same folder with the "transfer_train.py".
