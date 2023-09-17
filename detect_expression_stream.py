from pathlib import Path
import sys
import os
from numpy import random
import cv2
import torch
import numpy as np
import streamlit as st
import queue
import tkinter as tk
from tkinter import filedialog
import threading
import pandas as pd
import plotly.express as px
import time
from insightface.insight_face import iresnet50, iresnet18
from torchvision import transforms
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings(action='ignore', category=UserWarning, module='torch')
mpl.rcParams['font.family'] = 'STKAITI'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, vid_formats, LoadImages
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh, \
    increment_path, save_one_box
from utils.plots import plot_one_box

# 设置标题
st.set_page_config(page_title="课堂教学系统", layout="wide")
st.markdown("<h1 style='font-size: 38px;'>基于学生表情识别的课堂教学分析系统</h1>", unsafe_allow_html=True)

face_preprocess = transforms.Compose([
    transforms.ToTensor(),  # input PIL => (3,56,56), /255.0
    transforms.Resize((112, 112), antialias=True),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_feature(face_image, training=True):
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)

    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


# 表情识别
def emotion_classify(img):
    # 获得灰度图，并且在内存中创建一个图像对象
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(img, (48, 48))
    # 扩充维度，shape变为(1,48,48,1)
    # 将（1，48，48，1）转换成为(1,1,48,48)
    face = np.expand_dims(face, 0)
    face = np.expand_dims(face, 0)

    # 人脸数据归一化，将像素值从0-255映射到0-1之间
    face = face / 255.0
    new_face = torch.from_numpy(face)
    new_new_face = new_face.float().requires_grad_(False)
    new_new_face = new_new_face.to(device)

    # 调用我们训练好的表情识别模型，预测分类
    emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().cpu().numpy())
    return emotion_arg


# 对一帧图像进行处理（包括检测、识别和匹配），最后绘制条形图和折线图，返回这一帧中的人脸坐标、对应的表情和人脸ID
def detect_frame(model, im, im0s):
    # 图像预处理
    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference，预测
    pred = model(im)[0]

    # Apply NMS，非极大值抑制
    det = non_max_suppression_face(pred, 0.45, 0.45)[0]

    # Process detections，处理预测结果
    faces_xyxy = []
    faces_emotion = []
    faces_id = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
        Emocounter = [['confused', 0], ['happy', 0], ['neutral', 0], ['surprise', 0]]

        # start_time2 = time.perf_counter()
        # 逐个识别检测到的人脸
        for i in range(det.size()[0]):
            xyxy = det[i, :4].view(-1).tolist()
            faces_xyxy.append(xyxy)

            # 获取裁剪的人脸图像并进行表情识别
            img_emo = save_one_box(xyxy, im0s, BGR=True, save=False)
            emo_index = emotion_classify(img_emo)
            faces_emotion.append(emo_index)

            if faces_vector is not None:
                start_time2 = time.perf_counter()
                # 获取人脸特征，并匹配ID
                face_feature = get_feature(img_emo, training=True)
                scores = (face_feature @ faces_vector.T)
                ind = np.argmax(scores)
                if scores[ind] > 0.5:
                    im_name = faces_name[ind]
                else:
                    im_name = 'XX'
                end_time2 = time.perf_counter()
                faces_id.append(im_name)
                per_face_emos[im_name].append(emo_index)

            # 对应表情类别统计加1
            for j in range(4):
                if Emocounter[j][0] == emotion_labels[emo_index]:
                    Emocounter[j][1] += 1

        # 绘制柱状图, 配置相关参数
        dfe = pd.DataFrame()
        dfe['表情'] = ['困惑', '开心', '中性', '惊讶']
        dfe['数量'] = [Emocounter[0][1], Emocounter[1][1], Emocounter[2][1], Emocounter[3][1]]
        bar_chart = px.bar(dfe, x='表情', y='数量', text='数量',
                           color_discrete_sequence=['#F63366'] * len(dfe),
                           template='plotly_white')
        bar_placeholder.plotly_chart(bar_chart, use_container_width=True)

        # execution_time2 = end_time2 - start_time2
        # time2_placeholder.write("代码运行时间:识别匹配{}秒".format(execution_time2))

        if per_face_emos is not None:
            # 绘制一个学生的表情折线图
            y = per_face_emos[st.session_state.optionid]
            x = [i for i in range(len(y))]
            title = '学生' + st.session_state.optionid
            plt.plot(x, y, marker='o')
            plt.title(title)
            plt.ylabel('表情')  # 设置y轴标签
            plt.yticks([0, 1, 2, 3], dfe['表情'])
            with line_placeholder:
                st.pyplot(plt)
    return faces_xyxy, faces_emotion, faces_id


# 检测人脸
def rundetect(model, source, f_stride):
    imgsz = (640, 640)
    color = [[random.randint(0, 255) for _ in range(3)] for _ in range(4)]  #图像中标注框的颜色

    # Dataloader，加载带预测图片
    print('loading images', source)
    dataset = LoadImages(source, img_size=imgsz)

    # 选择性地对视频帧进行处理
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz[0], imgsz[0]).to(device).type_as(next(model.parameters())))  # run once
    flag = 0
    for _, im, im0s, vid_cap in dataset:  # 逐帧检测
        if flag % f_stride == 0:
            start_time = time.perf_counter()
            faces_xyxy, faces_emotion, faces_id = detect_frame(model, im, im0s)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            time_placeholder.write("代码运行时间:总{}秒".format(execution_time))

        if faces_xyxy:
            for i in range(len(faces_xyxy)):
                # 在原始图像上画框标定
                if faces_id:
                    label = faces_id[i] + ' ' + emotion_labels[faces_emotion[i]]
                else:
                    label = emotion_labels[faces_emotion[i]]
                plot_one_box(faces_xyxy[i], im0s, label=label, color=color[faces_emotion[i]])

        # 显示处理后的图像
        image_placeholder.image(im0s, channels='BGR')
        flag = flag + 1


def select_file():
    q = queue.Queue()

    def open_file_dialog():
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        file_path = filedialog.askopenfilename(filetypes=[('视频文件', ['*.mp4', '*.avi', 'wmv']), ('所有文件', '*.*')],
                                               title='选择文件')  # 显示文件选择对话框
        q.put(file_path)  # 将文件路径放入队列中
        root.destroy()  # 销毁Tkinter窗口

    threading.Thread(target=open_file_dialog).start()  # 在新线程中打开文件对话框
    selected_file = q.get()  # 从队列中获取文件路径
    return selected_file


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载人脸检测模型
    model = attempt_load('models/weights/yolov5m-face.pt', map_location=device)  # load FP32 model

    # 加载人脸匹配模型和人脸数据库
    weight = torch.load("insightface/resnet18_backbone.pth", map_location=device)
    model_emb = iresnet18()
    model_emb.load_state_dict(weight)
    model_emb.to(device)
    model_emb.eval()

    # 加载表情识别模型
    classification_model_path = 'model/weights/model_ResNet_CFER.pkl'
    classifier = torch.load(classification_model_path, map_location=device)

    return model, classifier, model_emb, device

@st.cache_resource
def faces_dataset(path):
    data_path = "data/features/" + path.split('/')[-1].split('.')[0] + ".npz"
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        # 学生的名字和特征向量，以及存储学生表情的变量
        name = data["name"].tolist()
        name.append("XX")
        vector = data["vector"]
        count = list(set(name))
        count.sort()
        return name, vector, count
    else:
        return None, None, []


emotion_labels = {0: 'confused', 1: 'happy', 2: 'neutral', 3: 'surprise'}  # 表情标签
detect_model, emotion_classifier, model_emb, device = load_model()

time_placeholder = st.empty()
time2_placeholder = st.empty()
# 创建放置图像、条形图、折线图和侧边栏上小组件的容器
col_image, col_bar = st.columns([0.7, 0.3])
with col_image:
    st.subheader('视频检测识别结果', divider='rainbow')
    image_placeholder = st.empty()
with col_bar:
    st.subheader('此时刻表情统计结果', divider='rainbow')
    bar_placeholder = st.empty()
col_line, _, _ = st.columns([0.5, 0.25, 0.25])
with col_line:
    st.subheader('学生的表情曲线', divider='rainbow')
    line_placeholder = st.empty()
form1 = st.sidebar.form('form1')
form2 = st.sidebar.form('form2')
form3 = st.sidebar.form('form3')

# 初始化会话状态中的值
if 'videofile' not in st.session_state:
    st.session_state.videofile = False
if 'f_stride' not in st.session_state:
    st.session_state.f_stride = 1
if 'optionid' not in st.session_state:
    st.session_state.optionid = '01'
all_name = []
faces_name = None
faces_vector = None

# 从侧边栏上传文件
with form1:
    st.write("请选择视频文件 [.MP4, .AVI, .WMV]")
    if st.form_submit_button('选择文件', help="若多次点击按钮，请先关闭已弹出的对话框，重新再点击一次"):
        st.session_state.videofile = select_file()
    if not st.session_state.videofile:
        file_path = "无"
    elif st.session_state.videofile.endswith(vid_formats):
        file_path = st.session_state.videofile
        faces_name, faces_vector, all_name = faces_dataset(file_path)
        if faces_name is not None:
            count = {i: faces_name.count(i) for i in faces_name}
            per_face_emos = {key: [] for key in count}
        else:
            per_face_emos = None
            form2.warning("无与视频名称相同的人脸数据库(.npz),无法进行身份识别")
    else:
        file_path = "请选择正确的视频文件格式！"
    st.write("文件:", file_path)

with form2:
    st.session_state.optionid = st.selectbox('请选择一个学生ID', all_name)
    st.session_state.f_stride = st.slider('检测帧间隔：', 1, 20)
    button = st.form_submit_button('确定')

# 执行目标检测
if st.session_state.videofile and st.session_state.videofile.split('.')[-1].lower() in vid_formats:
    rundetect(detect_model, st.session_state.videofile, st.session_state.f_stride)



