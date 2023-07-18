## Introduction

In this project, the grasp synthesis network is based on [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/INSTALL.md), and we use [GraspIt!](http://graspit-simulator.github.io/build/html/installation_linux.html) to view the results. Please configure the development environment according to the instructions in links.

Installation process:
* Download this code and unzip it. Note: 'cpp_wrappers', etc. are from [KPConv].

* [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/INSTALL.md): It is recommended to install the ubuntu version.

* [GraspIt!](http://graspit-simulator.github.io/build/html/installation_linux.html): It is recommended to install the ubuntu version.

* If your research continues to be based on the above two projects, please directly cite the original work.



## Grasp Synthesis
* In order to facilitate understanding, this code is based on the Barretthand with low degrees of freedom (the model and kinematics calculation are in './utils'). we also omit the pre-training process, etc., which require additional data generation and processing. If you are interested, you can add it by referring to the paper or contact us.

* Unzip './datasets/functional_area(0-2).zip' to the './datasets/functional_area/'.

* Run the following code to automatically process data and start network training :

          python train_GraspNet.py

* The results are in the folder './results', which can be viewed by directly using [GraspIt!], or copy all .xml files that you want to view to the 'your-graspit-root-dir/worlds/', and then run the following code :

          python show_result.py
