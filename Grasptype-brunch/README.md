## Introduction
This brunch is labeling the grasptype of each intances.
* In this project, we first obtain various grasp type from [Ganhand](https://github.com/enriccorona/GanHand) using MANO representation.
* Then we use the mapping method to obtain the dexterous hand keypoints.
* We design an simple MLP network to obtain the dexterous angles.
* We label the grasp type represented by the angles on the train set object, which is the extention of the hand-object representation based on [Toward-Human-Like-Grasp](https://github.com/zhutq-github/Toward-Human-Like-Grasp).
* Finally we train the classification network using the modified pointnet network, the reason is to generate the grasp type (represented by angles of dexterous hand) on test set objects. The classification accuarcy is 95%. Run the following code:

          python train_classify_grasp.py



