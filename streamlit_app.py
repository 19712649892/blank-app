import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2

def main():
    try:
        with open("instructions.md", "r", encoding="utf-8") as f:
            readme_text = st.markdown(f.read())
    except FileNotFoundError:
        readme_text = st.markdown("# 自动驾驶目标检测演示系统\n\n请在侧边栏选择“程序运行”开始体验")

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

#模式选择器
    st.sidebar.title("请选择以下选项")
    app_mode = st.sidebar.selectbox("",
        ["网页说明", "程序运行"])
    if app_mode == "网页说明":
        st.sidebar.success('请选择“程序运行”开始体验')
    elif app_mode == "程序运行":
        readme_text.empty()
        run_the_app()
# Streamlit动画功能
def download_file(file_path):
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # 动画化两个视觉元素的控制柄
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("正在下载 %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # 通过覆盖元素实现动画
                    weights_warning.warning(f"正在下载 {file_path}... ({counter / MEGABYTES:6.2f}/{length / MEGABYTES:6.2f} MB)")
                    progress_bar.progress(min(counter / length, 1.0))

    # 通过空数组函数.empty().移除视觉元素
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

#界面运行主方法
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache_data
    def load_metadata(url):
        return pd.read_csv(url)

    # 利用Pandas汇总元数据数据框
    @st.cache_data
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_biker": "biker",
            "label_car": "car",
            "label_pedestrian": "pedestrian",
            "label_trafficLight": "traffic light",
            "label_truck": "truck"
        })
        return summary

    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    summary = create_summary(metadata)

    #绘制用户界面元素以搜索目标物体
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("没有符合条件的图像帧，请选择不同的目标类别或数量范围。")
        return

    # 绘制用户界面元素以选择YOLO目标检测器的参数
    confidence_threshold, overlap_threshold = object_detector_ui()

    #从S3加载图像
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    image = load_image(image_url)

    #在图像上为地面对象添加框线
    boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    draw_image_with_boxes(image, boxes, "人工标注（真实值）",
        "**人工标注数据** (帧序号 `%i`)" % selected_frame_index)

    yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)
    draw_image_with_boxes(image, yolo_boxes, "实时计算机视觉（YOLOv3）",
        "**YOLOv3模型检测结果** (重叠阈值 `%3.1f`) (置信度阈值 `%3.1f`)" % (overlap_threshold, confidence_threshold))

# 查找对象类型的选择
def frame_selector_ui(summary):
    st.sidebar.markdown("# 图像帧选择")

    # 英文到中文映射
    label_map = {
        'biker': '骑行者',
        'car': '汽车',
        'pedestrian': '行人',
        'traffic light': '交通灯',
        'truck': '卡车'
    }
    chinese_labels = [label_map[col] for col in summary.columns]

    selected_chinese = st.sidebar.selectbox("搜索哪类目标？", chinese_labels, index=2)
    # 反向映射回英文
    object_type = {v: k for k, v in label_map.items()}[selected_chinese]

    min_elts, max_elts = st.sidebar.slider(f"选择 {selected_chinese} 的数量范围", 0, 25, [10, 20])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None

    # 从已选帧中选取一个帧
    selected_frame_index = st.sidebar.slider("选择图像帧序号", 0, len(selected_frames) - 1, 0)

    # 绘制altair图标
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x="selected_frame")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

#根据侧边栏中的内容选择框架
@st.cache_data(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

#参数选择
def object_detector_ui():
    st.sidebar.markdown("# 模型参数设置")
    confidence_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("重叠度阈值（IoU）", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

#绘制图像时叠加方框
def draw_image_with_boxes(image, boxes, header, description):
    LABEL_COLORS = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255],
    }
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    #绘制标题与图像
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

#文件下载
@st.cache_resource(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

#st.cache缓存机制以便重复使用图像
@st.cache_data(show_spinner=False)
def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

#运用YOLO检测图像
def yolo_v3(image, confidence_threshold, overlap_threshold):
    #加载网络
    @st.cache_resource
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    #运行YOLO神经网络
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    #重叠度阈值过高或置信度阈值过低时抑制结果
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: 'pedestrian',
        1: 'biker',
        2: 'car',
        3: 'biker',
        5: 'truck',
        7: 'truck',
        9: 'trafficLight'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

# 指向Streamlit公共S3存储桶的路径
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# 需要下载的外部文件
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

if __name__ == "__main__":
    main()