This project is related to the paper: DACrack: Unsupervised Domain Adaptation for crack detection. 

datasets
--------------------------------------------------------------------------------
(1) G45: This dataset consists of 569 gray-scale images, each with a resolution of 512x512. It was collected by a professional vehicle on the G45 national highway in Wuhan. 
-Our self-made crack dataset
(2) CFD: All images of this dataset were collected with iPhone5 camera on the urban roads of Beijing. CFD consists of 118 color images, and the resolution is 480×320 pixels. 
-https://doi.org/10.1109/TITS.2016.2552248
(3) CrackFCN: The background of this dataset is of concrete material, and the crack pattern is simple. All 148 images is resized to 320x320 pixels that are closely acquired with sufficient crack pixels. 
-https://doi.org/10.1111/mice.12412
(4) CrackTunnel: This dataset consists of 919 color images of the concrete tunnel surface, each with the resolution of 256×256 pixels. It was collected by a single-lens digital camera in a tunnel in Huzhou City, Zhejiang Province, China. 
-https://doi.org/10.1016/j.conbuildmat.2019.117367

We rotate the training images in all datasets by 90 degrees, and mirror the original image and the rotated image to generate new training samples. During traing, we resize all the training images to the resolution of 512x512.


Baseline
--------------------------------------------------------------------------------
The baseline network of paper, consist of the baseline models trained with G45, CFD, and CrackFCN training samples(e.g. Baseline\ResnetTrain_G45_lradam\train\ckpt), and the cross-datatset test results(e.g. Baseline\ResnetTrain_G45_lradam\test\G45_test_CFD_lradam_binary). You can directly test the model with try_test.py. The settings are presented in the readme file of Baseline directory.

DACrack
-------------------------------------------------------------------------------------------------------
The DACrack network of paper, consist of the DACrack models(e.g. The Cross-Structure Experiments of Tunnel-to-Pavement (CrackTunnel to G45)  Adaptation model trained with G45 training images，G45 training ground truth and CrackTunnel training images. Model Dir: DACrack\checkpoints\Resnet_G45_CrackTunnel_FEM+FAM+OAM), and the test results.(e.g. DACrack\results\Resnet_G45_CrackTunnel_FEM+FAM+OAM). You can directly test the model with try_test_best.py. The settings are presented in the readme file of DAcrack directory.


   



