from flask import Flask, request, json
import config as CFG
import cv2

# 相机模块
import camera_op as Camera
import main_process as mp

# 传入__name__初始化一个Flask实例
app = Flask(__name__)
camera = Camera.CameraOperation


def init_camera():
    # 打开设备
    if not camera.open_device():
        camera.close_device()
        camera.exit(0)


# app.route装饰器映射URL和执行的函数。这个设置将根URL映射到了hello_world函数上
@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/post')
def post():
    if request.method == "POST":
        response_data = request.json.get('data')
        print("response_data:", response_data)

        # 获取图像. 独立线程?
        # frame = camera.grab_image()
        # if frame is not None:
        #     cv2.imwrite("frame.jpg", frame)

        json_result = mp.do_by_commandID(response_data[CFG.COMMAND_ID])
        return json.dumps(json_result)


if __name__ == '__main__':
    # 运行本项目，host=0.0.0.0可以让其他电脑也能访问到该网站，port指定访问的端口。
    # 默认的host是127.0.0.1，port为8888
    # init_camera()
    app.run(host='0.0.0.0', port=8888)
