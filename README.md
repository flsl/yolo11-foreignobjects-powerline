# 基于yolov11的输电线路巡检异物检测系统python源码+pytorch模型+评估指标曲线+精美GUI界面

【算法介绍】

在电力行业向智能化、高效化运维转型的关键时期，输电线异物检测已成为保障电网安全的核心挑战之一。输电线路周边可能出现的塑料薄膜、风筝线、广告布等异物，极易因风力、振动等因素缠绕导线，引发短路、闪络甚至火灾事故，对电网安全构成重大威胁。传统人工巡检依赖望远镜或地面观测，受地形限制难以覆盖高山、峡谷等复杂区域，且对微小异物的识别能力不足；早期基于规则匹配的图像算法，受异物材质反光特性、拍摄视角偏差及植被遮挡干扰，误报率高达30%以上，无法满足电网"零隐患"运维需求。因此，开发具备高精度、强适应性的输电线异物智能检测系统，成为提升电网安全韧性的关键技术突破点。

现有技术瓶颈显著：人工登塔巡检不仅效率低下（单日仅能完成2-3公里线路排查），且存在高空坠落、触电等重大安全风险；基于颜色分割的传统算法难以区分异物与自然背景（如白色塑料袋与云层、黑色飘带与树影），在阴雨、雾霾等低能见度环境下性能断崖式下降；传统目标检测模型对异物的形态多样性（如缠绕、悬垂、贴附）和尺度变化（从几厘米到数米）适应性差，小目标漏检率超过40%。

基于YOLOv11的输电线异物智能检测系统为电网运维带来革命性变革。该系统充分发挥YOLOv11的端到端实时检测优势，结合电网场景特性进行深度优化，为新型电力系统建设提供了坚实的技术支撑。

【效果展示】

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/b29e3612565d40baab2ccf117c35e523.png"></div>

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/2010cbd749c44502ab92703325c297e2.png">&nbsp;</div>

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/619b97bf924c4abf823509dd4bb0884a.png">&nbsp;</div>

【测试环境】

windows10
anaconda3+python3.8
torch==2.3.0
ultralytics==8.3.81

【模型可以检测出类别】

balloon
kite
nest
trash

【训练数据集介绍】

数据集中有部分增强图片主要为旋转增强

数据集格式：Pascal VOC格式+YOLO格式(不包含分割路径的txt文件，仅仅包含jpg图片以及对应的VOC格式xml文件和yolo格式txt文件)
图片数量(jpg文件个数)：1544
标注数量(xml文件个数)：1544
标注数量(txt文件个数)：1544
标注类别数：4
标注类别名称(注意yolo格式类别顺序不和这个对应，而以labels文件夹classes.txt为准):["balloon","kite","nest","trash"]
每个类别标注的框数：
balloon 框数 = 240
kite 框数 = 344
nest 框数 = 927
trash 框数 = 67
总框数：1578
使用标注工具：labelImg
标注规则：对类别进行画矩形框
重要说明：暂无
特别声明：本数据集不对训练的模型或者权重文件精度作任何保证，数据集只提供准确且合理标注

图片预览：

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/09fbde6c3d394bdda012bdf16df41046.jpeg"></div>

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/436667de57ff4457939b5d4227c256e3.jpeg">&nbsp;</div>

标注例子：

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/7c6693295089419bb46543ba41cd971c.jpeg"></div>

【训练信息】

| 参数 | 值 |
|:---:|:---:|
| 训练集图片数 | 1466 |
| 验证集图片数 | 78 |
| 训练map | 98.3% |
| 训练精度(Precision) | 92.8% |
| 训练召回率(Recall) | 95.1% |

【验证集精度统计】

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| all | 78 | 78 | 0.928 | 0.951 | 0.983 | 0.63 |
| balloon | 8 | 8 | 0.841 | 1 | 0.995 | 0.832 |
| kite | 14 | 14 | 0.946 | 1 | 0.995 | 0.634 |
| nest | 53 | 53 | 0.927 | 0.925 | 0.948 | 0.499 |
| trash | 3 | 3 | 1 | 0.88 | 0.995 | 0.554 |

【界面设计】

```
class Ui_MainWindow(QtWidgets.QMainWindow):
    signal = QtCore.pyqtSignal(str, str)
 
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1280, 728)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
 
        self.weights_dir = './weights'
 
        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))
        self.picture.setStyleSheet("background:black")
        self.picture.setObjectName("picture")
        self.picture.setScaledContents(True)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))
        self.label_2.setObjectName("label_2")
        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)
        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))
        self.cb_weights.setObjectName("cb_weights")
        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)
 
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))
        self.label_3.setObjectName("label_3")
        self.hs_conf = QtWidgets.QSlider(self.centralwidget)
        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 181, 22))
        self.hs_conf.setProperty("value", 25)
        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)
        self.hs_conf.setObjectName("hs_conf")
        self.hs_conf.valueChanged.connect(self.conf_change)
        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_conf.setGeometry(QtCore.QRect(200, 100, 51, 22))
        self.dsb_conf.setMaximum(1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setProperty("value", 0.25)
        self.dsb_conf.setObjectName("dsb_conf")
        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)
        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_iou.setGeometry(QtCore.QRect(200, 160, 51, 22))
        self.dsb_iou.setMaximum(1.0)
        self.dsb_iou.setSingleStep(0.01)
        self.dsb_iou.setProperty("value", 0.45)
        self.dsb_iou.setObjectName("dsb_iou")
        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)
        self.hs_iou = QtWidgets.QSlider(self.centralwidget)
        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 181, 22))
        self.hs_iou.setProperty("value", 45)
        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)
        self.hs_iou.setObjectName("hs_iou")
        self.hs_iou.valueChanged.connect(self.iou_change)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 72, 21))
        self.label_5.setObjectName("label_5")
        self.le_res = QtWidgets.QTextEdit(self.centralwidget)
        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 400))
        self.le_res.setObjectName("le_res")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionopenpic = QtWidgets.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopenpic.setIcon(icon)
        self.actionopenpic.setObjectName("actionopenpic")
        self.actionopenpic.triggered.connect(self.open_image)
        self.action = QtWidgets.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action.setIcon(icon1)
        self.action.setObjectName("action")
        self.action.triggered.connect(self.open_video)
        self.action_2 = QtWidgets.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_2.setIcon(icon2)
        self.action_2.setObjectName("action_2")
        self.action_2.triggered.connect(self.open_camera)
 
        self.actionexit = QtWidgets.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionexit.setIcon(icon3)
        self.actionexit.setObjectName("actionexit")
        self.actionexit.triggered.connect(self.exit)
 
        self.toolBar.addAction(self.actionopenpic)
        self.toolBar.addAction(self.action)
        self.toolBar.addAction(self.action_2)
        self.toolBar.addAction(self.actionexit)
 
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)
        self.init_all()
```

【常用评估参数介绍】

在目标检测任务中，评估模型的性能是至关重要的。你提到的几个术语是评估模型性能的常用指标。下面是对这些术语的详细解释：

1.  **Class** ：

   - 这通常指的是模型被设计用来检测的目标类别。例如，一个模型可能被训练来检测车辆、行人或动物等不同类别的对象。

2.  **Images** ：

   - 表示验证集中的图片数量。验证集是用来评估模型性能的数据集，与训练集分开，以确保评估结果的公正性。

3.  **Instances** ：

   - 在所有图片中目标对象的总数。这包括了所有类别对象的总和，例如，如果验证集包含100张图片，每张图片平均有5个目标对象，则Instances为500。

4.  **P（精确度Precision）** ：

   - 精确度是模型预测为正样本的实例中，真正为正样本的比例。计算公式为：Precision = TP / (TP + FP)，其中TP表示真正例（True Positives），FP表示假正例（False Positives）。

5.  **R（召回率Recall）** ：

   - 召回率是所有真正的正样本中被模型正确预测为正样本的比例。计算公式为：Recall = TP / (TP + FN)，其中FN表示假负例（False Negatives）。

6.  **mAP50** ：

   - 表示在IoU（交并比）阈值为0.5时的平均精度（mean Average Precision）。IoU是衡量预测框和真实框重叠程度的指标。mAP是一个综合指标，考虑了精确度和召回率，用于评估模型在不同召回率水平上的性能。在IoU=0.5时，如果预测框与真实框的重叠程度达到或超过50%，则认为该预测是正确的。

7.  **mAP50-95** ：

   - 表示在IoU从0.5到0.95（间隔0.05）的范围内，模型的平均精度。这是一个更严格的评估标准，要求预测框与真实框的重叠程度更高。在目标检测任务中，更高的IoU阈值意味着模型需要更准确地定位目标对象。mAP50-95的计算考虑了从宽松到严格的多个IoU阈值，因此能够更全面地评估模型的性能。

这些指标共同构成了评估目标检测模型性能的重要框架。通过比较不同模型在这些指标上的表现，可以判断哪个模型在实际应用中可能更有效。

【使用步骤】

使用步骤：
（1）首先根据官方框架ultralytics安装教程安装好yolov11环境，并安装好pyqt5
（2）切换到自己安装的yolo11环境后，并切换到源码目录，执行python main.py即可运行启动界面，进行相应的操作即可

【提供文件】

python源码
yolo11n.pt模型
训练的map,P,R曲线图(在weights\results.png)
测试图片（在test_img文件夹下面）

注意提供训练的数据集，请到mytxt.txt文件中找到地址
<br>项目源码地址：https://mbd.pub/o/bread/YZWTlJprag==
