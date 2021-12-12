# CV Object and Lane Detection
 Hands on tutorial showing how to run Tensorflow Lite object detection models with hough lane detection.

## Step 1
First, let's make sure the camera interface is enabled in the Raspberry Pi Configuration Menu. Click the Pi icon in the top left corner of the screen, select Preferences -> Raspberry Pi Configuration, and go to the Interfaces tab and verify Camera is set to Enabled. If it isn't, enable it now, and reboot the Raspberry Pi.

## Step 2
Next, Open a terminal window and clone this GitHub repository by issuing the following command.
```
git clone https://github.com/kevinmuscara/CV-Object-and-Lane-Detection.git
```

This downloads all the required files and source code needed for this project into a folder called `CV-Object-and-Lane-Detection`, To rename it, issue the following commands.

```
mv CV-Object-and-Lane-Detection cv
cd cv
```

## Step 3
We will work in this directory for the remaining portions of this guide. Next, we have to create a virtual environment. 
A virtual environment is used to prevent any conflicts between packages or libraries that are already installed on your device. 

Install `virtualenv` by issuing:

```
sudo pip3 install virtualenv
```

Then, create your virtual environment by issuing:

```
python3 -m venv cv1-env
```

This will create a folder called `cv1-env` inside the project directory. This folder will hold all packages and libraries for this environment.
Next, we'll need to activate the virtual environment by issuing:

```
source cv1-env/bin/activate
```

Anytime you close or open the terminal window, you'll need to reissue the activation command from inside the `/home/pi/cv` directory to reactivate the environment.

## Step 4

Next, we have to install TensorFlow, OpenCV, and all dependencies needed for both packages. To install the needed libraries, issue the following:

```
bash install.sh
```

Once all the libraries and packages finish installing, you're ready to go! 

## Object Detection on webcam feed.

Run the following command to start the webcam feed.
```
python3 webcam.py
```

If you want to mess around with the threshold parameters that determine what objects get detected for a certain percentage, run the following:
```
python3 webcam.py --threshold=0.5
```
`0.5` is the same as 50%, meaning objects will only be detected if the program is 50% sure of what the object is.

To quit the program at anytime, press the `q` key.

## Object Detection and lane detection with dashcam footage.

AI is very rapidly taking over, one of the most well known AI projects are self driving cars. There's a lot that goes into computer vision for a self driving car, but this is a very simplistic example of a self driving cars computer vision.

To run the program issue the following:

```
python3 car.py
```

Just like with the webcam object detection, if you want to mess around with the threshold parameters, issue the following:

```
python3 car.py --threshold=0.5
```

To quit the program at anytime, press the `q` key.