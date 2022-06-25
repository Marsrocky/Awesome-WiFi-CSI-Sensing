[![GitHub](https://img.shields.io/github/license/Marsrocky/Awesome-WiFi-CSI-Sensing?color=blue)](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing/blob/main/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing/graphs/commit-activity)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

# Awesome WiFi Sensing
A list of awesome papers and cool resources on WiFi CSI sensing. Link to the *Github* if available is also present. 

*You are very welcome to suggest resources via pull requests*.

# Table of Contents
- [Benchmark](#benchmark)
- [Papers](#papers)
  - [Methods](#methods)
  - [Surveys](#surveys)
  - [Applications](#applications)
    - [Occupancy Detection](#occupancy-detection)
    - [Human Activity Recognition](#human-activity-recognition)
    - [Human Identification](#human-identification)
    - [Crowd Counting](#crowd-counting)
    - [Gesture Recognition](#gesture-recognition)
    - [Fall Detection](#fall-detection)
    - [Vital Sign Detection & Healthcare](#vital-sign-detection--healthcare)
    - [In-Car Activity Recognition](#in-car-activity-recognition)
    - [Pose Estimation](#pose-estimation)
    - [Indoor Localization](#indoor-localization)
  - [Challenges for Real-World Large-Scale WiFi Sensing](#challenges-for-real-world-large-scale-wifi-sensing)
    - [IoT System Design](#iot-system-design)
    - [Efficiency and Security](#efficiency-and-security)
    - [Cross-Environment WiFi Sensing](#cross-environment-wifi-sensing)
    - [Multi-modal Sensing (WiFi+CV/Radar)](#multi-modal-sensing-wificvradar)
- [Platforms](#platforms)
  - [CSI Tool](#csi-tool)
- [Datasets](#datasets)
- [Libraries & Codes](#libraries--codes)
  - [Libraries](#libraries)
  - [Github Repositories](#github-repositories)
    - [From Papers](#from-papers)
    - [From Developers](#from-developers)
- [Book Chapter](#book-chapter)

# Benchmark
* [Deep Learning and Its Applications in WiFi CSI Sensing]() | [[Github]]() 
  <br>A comprehensive benchmarking for deep learning models in WiFi sensing.

<!-- *********************************************************************** -->

# Papers

Papers are ordered by theme and inside each theme by publication date (submission date for arXiv papers).

## Methods
WiFi CSI sensing methods have enabled many applications, which can be divided into three categories:
* **Learning-based methods** learn the mapping functions from CSI data to the corresponding labels by [machine learning]() and deep learning.
* **Modeling-based methods** are based on physical theories like the [Fresnel Zone model](https://ieeexplore.ieee.org/abstract/document/8067692), or statistical models like the [Rician fading model](https://ieeexplore.ieee.org/abstract/document/9385792).
* **Hybrid methods** derive the strengths from learning-based and modeling-based methods.

## Surveys
* [WiFi Sensing with Channel State Information: A Survey](https://www.cs.wm.edu/~yma/files/WiFiSensing_YongsenMa_authorversion.pdf) ACM Computing Surveys (2019)
* [Device-Free WiFi Human Sensing: From Pattern-Based to Model-Based Approaches](https://ieeexplore.ieee.org/abstract/document/8067692) IEEE Communications Magazine (2017)
* [From RSSI to CSI: Indoor Localization via Channel Response](https://dl.acm.org/doi/abs/10.1145/2543581.2543592) ACM Computing Surveys (2013)

## Applications
### Occupancy Detection
* [Intelligent Wi-Fi Based Child Presence Detection System](https://ieeexplore.ieee.org/document/9747420) IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2022)
* [Machine Learning empowered Occupancy Sensing for Smart Buildings](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/icml2019/3/paper.pdf) IEEE ICML Climate Change Workshop (2019)
* [FreeDetector: Device-Free Occupancy Detection with Commodity WiFi](https://ieeexplore.ieee.org/abstract/document/8011040) IEEE International Conference on Sensing, Communication and Networking (SECON Workshops) (2017)

### Human Activity Recognition
* [Wi-Fi-Based Location-Independent Human Activity Recognition with Attention Mechanism Enhanced Method](https://www.mdpi.com/2079-9292/11/4/642/pdf) Electronics (2022)
* [Multimodal CSI-based Human Activity Recognition using GANs](https://ieeexplore.ieee.org/document/9431203) IEEE Internet of Things Journal (2021)
* [Two-Stream Convolution Augmented Transformer for Human Activity Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/16103) | [[Github](https://github.com/windofshadow/THAT)] AAAI Conference on Artificial Intelligence (AAAI) (2021)
* [Improving WiFi-based Human Activity Recognition with Adaptive Initial State via One-shot Learning](https://ieeexplore.ieee.org/abstract/document/9417590) IEEE Wireless Communications and Networking Conference (2021)
* [Data Augmentation and Dense-LSTM for Human Activity Recognition Using WiFi Signal](https://ieeexplore.ieee.org/abstract/document/9205901) IEEE Internet of Things Journal (2021)
* [DeepSeg: Deep-Learning-Based Activity Segmentation Framework for Activity Recognition Using WiFi](https://ieeexplore.ieee.org/document/9235578) | [[Github]](https://github.com/ChunjingXiao/DeepSeg) IEEE Internet of Things Journal (2021)
* [Robust CSI-based Human Activity Recognition using Roaming Generator](https://ieeexplore.ieee.org/abstract/document/9305332) IEEE International Conference on Control and Automation (ICCA) (2020)
* [CsiGAN: Robust Channel State Information-Based Activity Recognition With GANs](https://ieeexplore.ieee.org/document/8808929) | [[Github]](https://github.com/ChunjingXiao/CsiGAN) IEEE Internet of Things Journal (2019)
* [BeSense: Leveraging WiFi Channel Data and Computational Intelligence for Behavior Analysis](https://ieeexplore.ieee.org/abstract/document/8870275) IEEE Computational Intelligence Magazine (2019)
* [WiFi CSI Based Passive Human Activity Recognition Using Attention Based BLSTM](https://ieeexplore.ieee.org/abstract/document/8514811) IEEE Transactions on Mobile Computing (2019)
* [Deep Learning Networks for Human Activity Recognition with CSI Correlation Feature Extraction](https://ieeexplore.ieee.org/abstract/document/8761445) IEEE International Conference on Communications (ICC) (2019)
* [Device-free Occupancy Sensing Platform using WiFi-enabled IoT Devices for Smart Homes](https://ieeexplore.ieee.org/abstract/document/8391737) IEEE Internet of Things Journal (2018)
* [DeepSense: Device-Free Human Activity Recognition via Autoencoder Long-Term Recurrent Convolutional Network](https://ieeexplore.ieee.org/abstract/document/8422895) IEEE International Conference on Communications (ICC) (2018)
* [CareFi: Sedentary Behavior Monitoring System via Commodity WiFi Infrastructures](https://ieeexplore.ieee.org/document/8354831) IEEE Transactions on Vehicular Technology (2018)
* [Towards Occupant Activity Driven Smart Buildings via WiFi-enabled IoT Devices and Deep Learning](https://www.sciencedirect.com/science/article/abs/pii/S037877881831329X) Energy and Building (2018)
* [Poster:WiFi-based Device-free Human Activity Recognition via Automatic Representation Learning](https://www.researchgate.net/publication/320222015_Poster_WiFi-based_Device-Free_Human_Activity_Recognition_via_Automatic_Representation_Learning) Annual International Conference on Mobile Computing, MOBICOM-17 (2017)
* [Understanding and Modeling of WiFi Signal Based Human Activity Recognition](https://dl.acm.org/doi/abs/10.1145/2789168.2790093) Annual International Conference on Mobile Computing (MOBICOM) (2017) 

### Human Identification

* [CAUTION: A Robust WiFi-based Human Authentication System via Few-shot Open-set Gait Recognition](https://ieeexplore.ieee.org/abstract/document/9726794/) IEEE Internet of Things Journal (2022)
* [WiONE: One-Shot Learning for Environment-Robust Device-Free User Authentication via Commodity Wi-Fi in Man–Machine System](https://ieeexplore.ieee.org/abstract/document/9385792) IEEE Transactions on Computational Social Systems (2021)
* [Wifi-based Human Identification via Convex Tensor Shapelet Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11497) AAAI Conference on Artificial Intelligence AAAI-18 (2018)
* [NeuralWave: Gait-Based User Identification Through Commodity WiFi and Deep Learning](https://ieeexplore.ieee.org/document/8591820) | [[Github]](https://github.com/kdkalvik/WiFi-user-recognition) Annual Conference of the IEEE Industrial Electronics Society (IECON) (2018)
* [Non-Intrusive Biometric Identification for Personalized Computing Using Wireless Big Data](https://ieeexplore.ieee.org/document/8560141) | [[Github]](https://github.com/mobinets/wifiwalker) IEEE SmartWorld, Ubiquitous Intelligence & Computing, Advanced & Trusted Computing, Scalable Computing & Communications, Cloud & Big Data Computing, Internet of People and Smart City Innovation (2018)
* [WFID: Passive Device-free Human Identification Using WiFi Signal](https://dl.acm.org/doi/abs/10.1145/2994374.2994377) International Conference on Mobile and Ubiquitous Systems: Computing, Networking and Services (MOBIQUITOUS) (2016)
* [WiWho: WiFi-Based Person Identification in Smart Spaces](https://ieeexplore.ieee.org/abstract/document/7460727) ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN) (2016)
* [WiFi-ID: Human Identification Using WiFi Signal](https://ieeexplore.ieee.org/abstract/document/7536315) International Conference on Distributed Computing in Sensor Systems (DCOSS) (2016)
 
### Crowd Counting
* [Passive People Counting using Commodity WiFi](https://ieeexplore.ieee.org/document/9221456) IEEE 6th World Forum on Internet of Things (WF-IoT) (2020)
* [Device-free Occupancy Detection and Crowd Counting in Smart Buildings with WiFi-enabled IoT](https://www.sciencedirect.com/science/article/abs/pii/S0378778817339336) Energy and Building (2018)
* [FreeCount: Device-Free Crowd Counting with Commodity WiFi](https://ieeexplore.ieee.org/abstract/document/8255034) IEEE Global Communications Conference (2017)
* [WiCount: A Deep Learning Approach for Crowd Counting Using WiFi Signals](https://ieeexplore.ieee.org/abstract/document/8367378) IEEE International Conference on Ubiquitous Computing and Communications (ISPA/IUCC) (2017)
* [A Trained-once Crowd Counting Method Using Differential WiFi Channel State Information](https://dl.acm.org/doi/abs/10.1145/2935651.2935657) International on Workshop on Physical Analytics (WPA) (2016)
* [Electronic frog eye: Counting crowd using WiFi](https://ieeexplore.ieee.org/abstract/document/6847958) IEEE Conference on Computer Communications (INFOCOM) (2014)

### Gesture Recognition
* [WiHF: Enable User Identified Gesture Recognition with WiFi](https://ieeexplore.ieee.org/abstract/document/9155539) IEEE Conference on Computer Communications (INFOCOM) (2020)
* [Learning Gestures from WiFi: A Siamese Recurrent Convolutional Architecture](https://ieeexplore.ieee.org/document/8839094) IEEE Internet of Things Journal (2019)
* [Robust WiFi-Enabled Device-Free Gesture Recognition via Unsupervised Adversarial Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8487345/) | [[Github]](https://github.com/Marsrocky/WiADG) IEEE International Conference on Computer Communication and Networks (ICCCN) (2018)
* [WiFi-enabled Device-free Gesture Recognition for Smart Home Automation](https://ieeexplore.ieee.org/abstract/document/8444331) IEEE International Conference on Control and Automation (ICCA) (2018)
* [SignFi: Sign Language Recognition Using WiFi](https://dl.acm.org/doi/10.1145/3191755) | [[Github]](https://github.com/yongsen/SignFi) ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (2018)
* [WiFinger: leveraging commodity WiFi for fine-grained finger gesture recognition](https://dl.acm.org/doi/abs/10.1145/2942358.2942393) ACM International Symposium on Mobile Ad Hoc Networking and Computing (MobiHoc) (2016)
* [WiGeR: WiFi-Based Gesture Recognition System](https://www.mdpi.com/2220-9964/5/6/92) ISPRS International Journal of Geo-Information (2016)
* [WiG: WiFi-Based Gesture Recognition System](https://ieeexplore.ieee.org/abstract/document/7288485) International Conference on Computer Communication and Networks (ICCCN) (2015)

### Fall Detection
* [DeFall: Environment-Independent Passive Fall Detection Using WiFi](https://ieeexplore.ieee.org/document/9552243) IEEE Internet of Things Journal (2022)
* [A WiFi-Based Smart Home Fall Detection System Using Recurrent Neural Network](https://ieeexplore.ieee.org/abstract/document/9186064) IEEE Transactions on Consumer Electronics (2020)
* [RT-Fall: A Real-Time and Contactless Fall Detection System with Commodity WiFi Devices](https://ieeexplore.ieee.org/abstract/document/7458198) IEEE Transactions on Mobile Computing (2017)
* [FallDeFi: Ubiquitous Fall Detection using Commodity Wi-Fi Devices](https://dl.acm.org/doi/abs/10.1145/3161183) the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (2017)

### Vital Sign Detection & Healthcare
* [Device-Free Multi-Person Respiration Monitoring Using WiFi](https://ieeexplore.ieee.org/abstract/document/9179993) IEEE Transactions on Vehicular Technology (2020)
* [Sleepy: Wireless Channel Data Driven Sleep Monitoring via Commodity WiFi Devices](https://ieeexplore.ieee.org/abstract/document/8399492) IEEE Transactions on Big Data (2020)
* [WiFi-Based Real-Time Breathing and Heart Rate Monitoring during Sleep](https://ieeexplore.ieee.org/abstract/document/9014297) IEEE Global Communications Conference (GLOBECOM) (2019)
* [Contactless Respiration Monitoring Via Off-the-Shelf WiFi Devices](https://ieeexplore.ieee.org/abstract/document/7345587) IEEE Transactions on Mobile Computing (2016)
* [Human respiration detection with commodity wifi devices: do user location and body orientation matter?](https://dl.acm.org/doi/abs/10.1145/2971648.2971744) Ubicomp (2016)

### In-Car Activity Recognition
* [CARIN: Wireless CSI-based Driver Activity Recognition under the Interference of Passengers](https://dl.acm.org/doi/abs/10.1145/3380992) ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (2020)
* [WiCAR: WiFi-based in-Car Activity Recognition with Multi-Adversarial Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9068635) IEEE/ACM 27th International Symposium on Quality of Service (IWQoS) (2019)
* [WiDriver: Driver Activity Recognition System Based on WiFi CSI](https://link.springer.com/article/10.1007/s10776-018-0389-0) International Journal of Wireless Information Networks (2018)

### Pose Estimation
* [3D Human Pose Estimation Using WiFi Signals](https://dl.acm.org/doi/abs/10.1145/3485730.3492871) ACM Conference on Embedded Networked Sensor Systems (SenSys) (2021)
* [Towards 3D human pose construction using wifi](https://dl.acm.org/doi/abs/10.1145/3372224.3380900) Annual International Conference on Mobile Computing and Networking (MOBICOM) (2020)
* [Person-in-WiFi: Fine-Grained Person Perception Using WiFi](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Person-in-WiFi_Fine-Grained_Person_Perception_Using_WiFi_ICCV_2019_paper.html) IEEE/CVF International Conference on Computer Vision (ICCV) (2019)

### Indoor Localization
* [RF Sensing in the Internet of Things: A General Deep Learning Framework](https://par.nsf.gov/servlets/purl/10087331) IEEE Communications Magazine (2018)
* [Passive Indoor Localization Based on CSI and Naive Bayes Classification](https://ieeexplore.ieee.org/abstract/document/7902212) IEEE Transactions on Systems, Man, and Cybernetics: Systems (2018)
* [CSI-Based Fingerprinting for Indoor Localization: A Deep Learning Approach](https://ieeexplore.ieee.org/abstract/document/7438932) IEEE Transactions on Vehicular Technology (2017)
* [CSI-Based Indoor Localization](https://ieeexplore.ieee.org/abstract/document/6244790) IEEE Transactions on Parallel and Distributed Systems (2013)

## Challenges for Real-World Large-Scale WiFi Sensing

### IoT System Design
* [Device-free Occupancy Sensing Platform using WiFi-enabled IoT Devices for Smart Homes](https://ieeexplore.ieee.org/abstract/document/8391737) IEEE Internet of Things Journal (2018)

### Efficiency and Security
* [EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression](https://ieeexplore.ieee.org/abstract/document/9667414) | [[Github]](https://github.com/Marsrocky/EfficientFi) IEEE Internet of Things Journal (2022) 
* [AutoFi: Towards Automatic WiFi Human Sensing via Geometric Self-Supervised Learning](https://arxiv.org/abs/2205.01629) arXiv:2205.01629 (2022)
* [RobustSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition](https://arxiv.org/abs/2204.01560) arXiv:2204.01560 (2022)
* [CSITime: Privacy-preserving human activity recognition using WiFi channel state information](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004391) Neural Networks, (2022)
* [WiFederated: Scalable WiFi Sensing using Edge Based Federated Learning](https://www.people.vcu.edu/~ebulut/FL-IoT-journal-2021.pdf) IEEE Internet of Things Journal (2021) 
* [A Lightweight Deep Learning Algorithm for WiFi-Based Identity Recognition](https://ieeexplore.ieee.org/abstract/document/9427070) IEEE Internet of Things Journal (2021)
* [An Experimental Study of CSI Management to Preserve Location Privacy](https://dl.acm.org/doi/10.1145/3411276.3412187) | [[Github]](https://github.com/seemoo-lab/csicloak) ACM International Workshop on Wireless Network Testbeds, Experimental evaluation & Characterization (WiNTECH) (2020)

### Cross-Environment WiFi Sensing
* [Privacy-Preserving Cross-Environment Human Activity Recognition](https://ieeexplore.ieee.org/abstract/document/9626548) IEEE Transactions on Cybernetics (2021)
* [Consensus Adversarial Domain Adaptation](https://ojs.aaai.org/index.php/AAAI/article/download/4552/4430) AAAI-19 (2019)
* [CrossSense: Towards Cross-Site and Large-Scale WiFi Sensing](https://dl.acm.org/doi/10.1145/3241539.3241570) Annual International Conference on Mobile Computing and Networking (MOBICOM) (2018)
* [Towards Environment Independent Device Free Human Activity Recognition](https://dl.acm.org/doi/abs/10.1145/3241539.3241548) Annual International Conference on Mobile Computing and Networking (MOBICOM) (2018)
* [Joint Adversarial Domain Adaptation for Resilient WiFi-Enabled Device-Free Gesture Recognition](https://ieeexplore.ieee.org/abstract/document/8614062) IEEE International Conference on Machine Learning and Applications (2018)
* [Fine-Grained Adaptive Location-Independent Activity Recognition using Commodity WiFi](https://ieeexplore.ieee.org/abstract/document/8377133) IEEE Wireless Communications and Networking Conference (WCNC) (2018)

### Multi-modal Sensing (WiFi+CV/Radar)
* [WiFE: WiFi and Vision based Intelligent Facial-Gesture Emotion Recognition](https://www.researchgate.net/profile/Xiang-Zhang-54/publication/340826489_WiFE_WiFi_and_Vision_based_Intelligent_Facial-Gesture_Emotion_Recognition/links/5ec72a1192851c11a87da07b/WiFE-WiFi-and-Vision-based-Intelligent-Facial-Gesture-Emotion-Recognition.pdf) arXiv:2004.09889 (2020)
* [WiFi and Vision Multimodal Learning for Accurate and Robust Device-Free Human Activity Recognition](https://openaccess.thecvf.com/content_CVPRW_2019/papers/MULA/Zou_WiFi_and_Vision_Multimodal_Learning_for_Accurate_and_Robust_Device-Free_CVPRW_2019_paper.pdf) IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2019)
* [Can WiFi Estimate Person Pose?](https://arxiv.org/abs/1904.00277) | [[Github]](https://github.com/geekfeiw/WiSPPN) arXiv:1904.00277 (2019)
* [XModal-ID: Using WiFi for Through-Wall Person Identification from Candidate Video Footage](https://dl.acm.org/doi/abs/10.1145/3300061.3345437) Annual International Conference on Mobile Computing and Networking (MOBICOM) (2019)
* [CSI-Net: Unified Human Body Characterization and Pose Recognition](https://arxiv.org/pdf/1810.03064.pdf) | [[Github]](https://github.com/geekfeiw/CSI-Net) arXiv:1810.03064 (2018)

<!-- *********************************************************************** -->

# Platforms

## CSI Tool
* [[Intel 5300 NIC]](https://dhalperi.github.io/linux-80211n-csitool/) Pioneered CSI tool enables Intel 5300 NIC to extract CSI data. It supports 30 subcarriers for a pair of antennas running on 20MHz.
* [[Atheros CSI Tool]](https://wands.sg/research/wifi/AtherosCSI/) Revamped CSI tool enables various Qualcomm Atheros NIC to extract CSI data. It supports 114 subcarriers for a pair of antennas running on 40MHz.
* [[Nexmon CSI Tool]](https://github.com/seemoo-lab/nexmon_csi) Mobile CSI Tool enables mobile phone and embedded device (RasPi) to extract CSI data of up to 256 subcarriers for a pair of antennas running on 80MHz.
* [[ESP32 CSI Tool]](https://stevenmhernandez.github.io/ESP32-CSI-Tool/) The ESP32 CSI Toolkit provides researchers access to Channel State Information (CSI) directly from the ESP32 microcontroller. 
* [[Software Defined Radio (SDR)]](https://www.ettus.com/) platforms, such as [Universal Software Radio Peripheral (USRP)](https://www.ettus.com/) and [Wireless Open Access Research Platform (WARP)](https://warpproject.org/trac), provide CSI measurements at 2.4GHz, 5GHz, and 60GHz.

# Datasets
* [[NTU-Fi]]() The NTU-Fi dataset is the only CSI dataset with 114 subcarriers per pair of antennas, collected using Atheros CSI tool. It consists of 6 human activities and 14 human gait patterns.
* [[Widar 3.0]](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset) The Widar 3.0 project is a large dataset designed for use in WiFi-based hand gesture recognition, collected using Intel 5300 NIC (30 subcarriers). The dataset consists of 258K instances of hand gestures with a duration of totally 8,620 minutes and from 75 domains.
* [[WiAR]](https://github.com/linteresa/WiAR) The WiAR dataset contains sixteen activities including coarse-grained activity and gestures performed by ten volunteers with 30 times every volunteer.
* [[UT-HAR]](https://github.com/ermongroup/Wifi_Activity_Recognition) The dataset is collected in ''A Survey on Behaviour Recognition Using WiFi Channel State Information''. It consists of continuous CSI data for 6 activities without golden segmentation timestamp for each sample.
* [[SignFi]](https://github.com/yongsen/SignFi) Channel State Information (CSI) traces for sign language recognition using WiFi.

<!-- *********************************************************************** -->

# Libraries & Codes

## Libraries
* [[SenseFi]]() Deep learning libraries for WiFi CSI sensing  (PyTorch) (Model Zoo)

## Github Repositories
### From Papers
* [EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression](https://ieeexplore.ieee.org/abstract/document/9667414) | [[Github]](https://github.com/Marsrocky/EfficientFi) (Python) (2022) 
* [DeepSeg: Deep-Learning-Based Activity Segmentation Framework for Activity Recognition Using WiFi](https://ieeexplore.ieee.org/document/9235578) | [[Github]](https://github.com/ChunjingXiao/DeepSeg) (Python) (2021)
* [Two-Stream Convolution Augmented Transformer for Human Activity Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/16103) | [[Github](https://github.com/windofshadow/THAT)] (Python) (2021)
* [An Experimental Study of CSI Management to Preserve Location Privacy](https://dl.acm.org/doi/10.1145/3411276.3412187) | [[Github]](https://github.com/seemoo-lab/csicloak) (Python) (2020)
* [CsiGAN: Robust Channel State Information-Based Activity Recognition With GANs](https://ieeexplore.ieee.org/document/8808929) | [[Github]](https://github.com/ChunjingXiao/CsiGAN) (Python) (2019)
* [Can WiFi Estimate Person Pose?](https://arxiv.org/abs/1904.00277) | [[Github]](https://github.com/geekfeiw/WiSPPN)
* [CSI-Net: Unified Human Body Characterization and Pose Recognition](https://arxiv.org/pdf/1810.03064.pdf) | [[Github]](https://github.com/geekfeiw/CSI-Net)
* [Robust WiFi-Enabled Device-Free Gesture Recognition via Unsupervised Adversarial Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8487345/) | [[Github]](https://github.com/Marsrocky/WiADG) (Python) (2018)
* [NeuralWave: Gait-Based User Identification Through Commodity WiFi and Deep Learning](https://ieeexplore.ieee.org/document/8591820) | [[Github]](https://github.com/kdkalvik/WiFi-user-recognition) (Python) (2018)
* [Non-Intrusive Biometric Identification for Personalized Computing Using Wireless Big Data](https://ieeexplore.ieee.org/document/8560141) | [[Github]](https://github.com/mobinets/wifiwalker) (MATLAB) (2018)
* [SignFi: Sign Language Recognition Using WiFi](https://dl.acm.org/doi/10.1145/3191755) | [[Github]](https://github.com/yongsen/SignFi) (MATLAB & Python) (2018)

### From Developers
* [Keystroke Recognition by Machine Learning and DTW](https://github.com/Ericfengdc/Keystroke-recognition-of-Smart-Device-Based-on-WIFI) (MATLAB & Python)
* [Gait Recognition by SVM](https://github.com/thinszx/WiFi-CSI-gait-recognition) (MATLAB)
* [BiLSTM for Human Activity Recognition](https://github.com/ludlows/CSI-Activity-Recognition) (Tensorflow 2.0)
* [LSTM for Human Activity Recognition](https://github.com/Retsediv/WIFI_CSI_based_HAR) (Pytorch)
* [SVM for Human Activity Recognition](https://github.com/noahcroit/HumanActivityDetection_using_CSI_MATLAB) (MATLAB)
* [Gesture Data Collection Tool for Nexmon CSI Tool](https://github.com/dingyiyi0226/gesture-recognition-csi) (Python)



# Book Chapter
* [Smart Wireless Sensing: From IoT to AIoT](https://link.springer.com/book/10.1007/978-981-16-5658-3) Springer (2021)
* [Deep Learning and Unsupervised Domain Adaptation for WiFi-based Sensing](https://www.worldscientific.com/doi/abs/10.1142/9789811218842_0004) "Generalization with Deep Learning", World Scientific (2021)
