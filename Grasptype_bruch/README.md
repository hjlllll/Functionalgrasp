## Introduction
This brunch is labeling the grasp type of each instance. First need to download in [dataset](https://drive.google.com/file/d/1VbnEJ7rNEAeAG8YJd7kF2I1SEHLdf7v0/view?usp=drive_link) and put in dataset_hjl/labeled_data/use/about_grasp/0.002_s420_465-100_ex-True_seg-True_train_record-is111111-new_withlabelgrasp-expanded-100-unsampled.pkl

* In this project, we first obtain 33 kinds of various grasp types from [Ganhand](https://github.com/enriccorona/GanHand) using MANO representation: taxonomy.npy
* Then we use the mapping method to obtain the dexterous hand key points using the method same as [TeachNet](https://github.com/Smilels/TeachNet_Teleoperation) if vis:
   
          cd grasptype_brunch
          python map_to_shadow.py


* We design a simple MLP network to obtain the dexterous angles, the dexterous hand grasp 33 taxonomy is in Functionalgrasp/result_xml/3grasptype :
 
          python train_ik_model.py

* We label two grasp types on each instance represented by the angles, which is the extension of the hand-object representation based on [Toward-Human-Like-Grasp](https://github.com/zhutq-github/Toward-Human-Like-Grasp).
* Finally we train the Multi-label classification network using the modified pointnet network, the reason is to generate the grasp type (represented by angles of dexterous hand) on test set objects. We obtain 0.8 accuracy on the test set to generate prior grasp type and write it into our new hand-object representation dataset in the dataset_hjl directory. Run the following code(train/test):

          python classify_grasp.py

* The pre-trained model available in directory ./checkpoint
## Requirements
- **MANO layer**: To obtain mano & manopth directory == Follow instructions from the MANO layer project in [here](https://raw.githubusercontent.com/hassony2/manopth). 
