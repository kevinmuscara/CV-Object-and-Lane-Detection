# CV Object and Lane Detection
This tutorial is designed to demonstrate how to run TensorFlow Lite object detection models with Hough Lane Detection on a Raspberry Pi. The tutorial is divded into several steps to ensure that all necessary dependencies are installed and configured properly. 

This tutorial was designed to be utilized in curriculum to introduce students to AI.

## Step 1: Enable the camera interface
* Open the Raspberry Pi Configuration Menu by clicking on the Pi icon in the top left corner of the screen.
* Select `Preferences -> Raspberry Pi Configuration`.
* Go to the `Interfaces` tab and verify that `Camera` is set to `Enabled`.
* If the camera interface is not enabled, enable it and reboot your Raspberry Pi.

## Step 2: Clone the GitHub repository
* Open a terminal window and clone the ncessary files and source code by issuing the following command:
```
git clone https://github.com/kevinmuscara/CV-Object-and-Lane-Detection.git
```

* This will download the files and source code needed for this project into a folder called `CV-Object-and-Lane-Detection`. To rename it, issue the following commands:

```
mv CV-Object-and-Lane-Detection cv
cd cv
```

## Step 3: Create a virtual environment
* A virtual environment is used to prevent any conflicts between packages or libraries that are already installed on the device. To create a virtual environment, first install `virtualenv` by issuing:

```
sudo pip3 install virtualenv
```

* Next, create the virtual environment by issuing:

```
python3 -m venv cv1-env
```
* This will create a folder called `cv1-env` inside the project directory. This folder will hold all packages and libraries for this environment. To activate the virtual environment, issue the following command:
```
source cv1-env/bin/activate
```
* Note that anytime you close or open the terminal window, you will need to reissue the activation command from inside the `/home/pi/cv` directory to reactivate the environment.

## Step 4: Install TensorFlow, OpenCV, and other dependencies
* To install the necessary libraries, issue the following command:
```
bash install.sh
```

## Step 5: Object Detection on webcam feed.
* To start the webcam feed, issue the following command:
```
python3 webcam.py
```

* To change the threshold parameters that determine what objects get detected for a certain percentage, issue the following command:
```
python3 webcam.py --threshold=0.5
```
* Note that `0.5` is the same as 50%, meaning objects will only be detected if the program is 50% certain of what the object is. Press the `q` key to quit the program at any time.

## Step 6: Object and Lane detection with dashcam footage.
* To run the program, issue the following command:
```
python3 car.py
```
* To change the threshold parameters, issue the following command:
```
python3 car.py --threshold=0.5
```
* Press the `q` key to quit the program at any time.

This tutorial provides a basic example of computer vision for self-driving cars by using object detection and lane detection. Keep in mind that this is a simplified example.
