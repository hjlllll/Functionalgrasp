## Introduction
This brunch is labeling the grasp type of each instances.
* In this project, we first obtain 33 kinds of various grasp types from [Ganhand](https://github.com/enriccorona/GanHand) using MANO representation: taxonomy.npy
* Then we use the mapping method to obtain the dexterous hand key points using the method same as [TeachNet](https://github.com/Smilels/TeachNet_Teleoperation) if vis:
   
          python map_to_shadow.py


* We design a simple MLP network to obtain the dexterous angles, the dexterous hand grasp 33 taxonomy is in Functionalgrasp/result_xml/3grasptype :
 
          python train_ik_model.py

* We label two grasp types on each instance represented by the angles, which is the extension of the hand-object representation based on [Toward-Human-Like-Grasp](https://github.com/zhutq-github/Toward-Human-Like-Grasp).
* Finally we train the Multi-label classification network using the modified pointnet network, the reason is to generate the grasp type (represented by angles of dexterous hand) on test set objects. We obtain 0.8 accuracy on the test set to generate prior grasp type and write it into our new hand-object representation dataset. Run the following code(train/test):

          python classify_grasp.py
