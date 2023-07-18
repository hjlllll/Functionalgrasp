## Grasp Synthesis
* In order to facilitate understanding, this code is based on the Barretthand with low degrees of freedom (the model and kinematics calculation are in './utils'). we also omit the pre-training process, etc., which require additional data generation and processing. If you are interested, you can add it by referring to the paper or contact us.

* Unzip './datasets/functional_area(0-2).zip' to the './datasets/functional_area/'.

* Run the following code to automatically process data and start network training :

          python train_GraspNet.py

* The results are in the folder './results', which can be viewed by directly using [GraspIt!], or copy all .xml files that you want to view to the 'your-graspit-root-dir/worlds/', and then run the following code :

          python show_result.py
