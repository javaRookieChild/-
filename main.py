import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from cv2 import getTickCount
import base64
import os

st.title("工业元器件状态检测")
st.header("欢迎使用")
st.subheader("选择图片、视频或摄像头进行状态检测")

def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

main_bg('./pics/background.png')

st.markdown("""  
**功能介绍**：  
- 支持图片检测功能。  
- 视频检测功能
- 摄像头检测功能
""")

# 侧边栏
st.sidebar.title("选择检测类型")
detection_type = st.sidebar.selectbox(
    "选择检测类型",
    ("图片检测", "视频检测", "实时摄像头检测")
)

# 加载YOLO模型
model = YOLO('../runs/detect/train/weights/best.pt')

# 图片检测
if detection_type == "图片检测":
    st.sidebar.markdown("**请选择要检测的图片**")
    with st.spinner('等待文件上传...'):
        file_pic = st.sidebar.file_uploader('')

    if file_pic is not None:
        pil_image = Image.open(file_pic)
        img = np.array(pil_image)
        result = model.predict(img)
        annotated_frame = result[0].plot()
        st.image(annotated_frame, channels="BGR", use_column_width=True)
        st.markdown(f"**检测结果**：已显示在上方图片中。")

elif detection_type == "视频检测":
    st.sidebar.markdown("**上传要检测的视频**")
    video_file = st.sidebar.file_uploader("上传视频文件", type=["mp4", "avi"])  # 假设支持mp4和avi格式
    if video_file is not None:
        # 将上传的视频文件保存为临时文件
        temp_video_path = "./temp_video." + video_file.name.split('.')[-1]
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        st.sidebar.text("视频上传成功，开始处理...")

        # 初始化视频捕获对象
        cap = cv2.VideoCapture(temp_video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            st.error("无法打开视频文件，请重试。")
        else:
            # 准备存放视频的HTML标签
            video_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 对当前帧进行目标检测
                results = model.predict(frame)
                annotated_frame = results[0].plot()

                # 转换颜色空间以适应Streamlit显示
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img_array = np.array(annotated_frame)

                # 显示检测结果
                video_placeholder.image(img_array, channels="RGB")

                # 可以在这里添加延时或根据需要控制处理速度

            cap.release()
            st.success("视频处理完成。")
            os.remove(temp_video_path)  # 处理完毕后删除临时文件


elif detection_type == "实时摄像头检测":
    st.sidebar.markdown("**开始实时摄像头检测**")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        loop_start = getTickCount()
        success, frame = cap.read()

        if success:
            height, width = frame.shape[:2]
            new_height = height // 3
            new_width = width // 3
            frame = cv2.resize(frame, (new_width, new_height))
            results = model.predict(source=frame)
            annotated_frame = results[0].plot()

            cv2.imshow('Real-time detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()