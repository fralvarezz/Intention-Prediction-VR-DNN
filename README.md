# Intention Prediction in Virtual Reality using Deep Neural Networks

## Installation

In order to use this application to its fullest a VR headset with eye tracking capabilities is required. We used HTC Vive Pro. Using another VR headset would require
additional development from the user to integrate eye tracking and VR features.

### Prerequesites

 - Unity 2020.3.20f1
 - SRanipal Runtime (1.3.2.0)
 - Python 3.9.0 or newer (Tested on 3.9.12)
 - Anaconda3
 - SteamVR (latest version)

#### Python

We used Anaconda to run the python scripts and the installation can be replicated by following these steps: 

``` $ conda create --name myenv ```

``` $ conda activate myenv ```


In order to install the required Python packages the following command can be ran from inside the **Thesis/Assets/Python Scripts** folder:

``` $ pip install -r requirements.txt ```

Finally, to install the PyTorch package with CUDA capabilties:

``` $ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch ```

## Executing the application

### Training

To train a model data first has to be collected. First open the Unity scene "ShelfOfItems" and make sure that the ExperimentManager game object is activated.

![image](https://user-images.githubusercontent.com/22989470/171205307-3b349495-6b7a-44c6-bfb0-5624db61d98a.png)

It is also important to make sure the EyeLogger component state is set to Logging

![image](https://user-images.githubusercontent.com/22989470/171256352-b46fb289-524b-4330-9e5a-c6f5e9fcf515.png)



When running the scene the application will start logging the participant behavior to a file called eyeLog_x, where x is sequential.

After data has been collected it can be used to train a model using the python script 

``` $ python LSTM_NN.py ```

Make sure that the line:

``` $ up = UNITY_CSV_PARSER.UnityParser(<PATH TO FILE>) ```

is pointing to the collected data. After the training is finished a model will be placed inside the NN_Models folder.

### Inference

The model can then be used for inference by running the network inference script:

``` $ python NetworkSock.py ```

Its again important that the correct model is used. This can be set on line 87 of the NetworkSock.py script:

``` self.ort_sess = ort.InferenceSession(<PATH TO FILE>) ```

The NetworkSock.py will listen for clients to run inference on localhost:18500.

You can then configure the Unity application for iference mode by first disabling the ExperimentManager and enabling the NetworkInference game object.

![image](https://user-images.githubusercontent.com/22989470/171255812-4f61af1c-2b89-4a60-839c-56291e819023.png)

Make sure that the EyeLogger component state is set to inference.

![image](https://user-images.githubusercontent.com/22989470/171255977-70d0f566-fc3b-47a8-ab92-4b51d55e5de7.png)

The application will then be able to predict user intent!

**Note:**  It is not necessary to train your own model. The git repository contains a pretrained model so that inference can be run right away.
