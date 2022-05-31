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

### Training

To train a model data first has to be collected. First open the Unity scene "ShelfOfItems" and make sure that the ExperimentManager game object is activated and that
the EyeLogger component state variable is set to Logging.

When running the scene the application will start logging the participant behavior to a file called eyeLog_x, where x is sequential.

After data has been collected it can be used to train a model using the python script 



![image](https://user-images.githubusercontent.com/22989470/171205307-3b349495-6b7a-44c6-bfb0-5624db61d98a.png)

