## Introduction
This brunch is labeling the grasp type of each instance. First need to download in [dataset](https://drive.google.com/file/d/1hhPGqu1B71E85wF71elP_e0D0Fydv0ED/view?usp=drive_link) and put in Data/labeled_data/use/about_grasp/0.002_s420_465-100_ex-True_seg-True_train_record-is111111-new_withlabelgrasp-expanded-100-unsampled.pkl

## Grasp Synthesis
* Run the following code to automatically process data and start network training :

          python train_GraspNet.py

* The results are in the folder './results', which can be viewed by directly using [GraspIt!], or copy all .xml files that you want to view to the 'your-graspit-root-dir/worlds/', and then run the following code :

          python utils/show_result.py
