# Online People Counter
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Version: Python](https://img.shields.io/badge/Python-3.7.2-blue)](https://www.python.org/downloads/) [![Version: Flask](https://img.shields.io/badge/Flask-1.1.X-blue)](https://flask.palletsprojects.com/en/1.1.x/) [![Version: Flask](https://img.shields.io/badge/Yolo-V3-blue)](https://pjreddie.com/darknet/yolo/)

<p align="center">
![Detection](https://media.giphy.com/media/dv1V127SvGgn2e908G/giphy.gif)
</p>


Online people counter using YoloV3 and Flask. I used Heroku to send to production: [Click here to have a demo.](http://count-people.herokuapp.com/). It reads a image file sent by the user or frames from an online feed. 

## Technologies :rocket: :

  * [Python](https://reactjs.org/)
  * [Flask](https://flask.palletsprojects.com/en/1.1.x/)
  * [YoloV3](https://pjreddie.com/darknet/yolo/)

## Setup
Just clone the repo:
```sh
git clone https://github.com/iurymelo/online-people-counter 
cd online-people-counter 
```
Install the requirements:
```sh
pip install -r requirements.txt
```
Download the YoloV3 weights [BY CLICKING HERE](https://pjreddie.com/media/files/yolov3.weights) and save it on dnn_config_files.

Now just run Flask app:
```
python app.py
```
The service will be listening in
http://127.0.0.1:5000

**Made by Iury Melo**
