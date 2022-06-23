[![GitHub](https://img.shields.io/github/license/Marsrocky/Awesome-WiFi-CSI-Sensing?color=blue)](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing/blob/main/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing/graphs/commit-activity)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

# Awesome WiFi Sensing
A list of awesome papers and cool resources on WiFi CSI sensing. Link to the code if available is also present. 

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
    - [Fall Detection (for Healthcare)](#fall-detection-for-healthcare)
    - [Respiration Detection](#respiration-detection)
    - [Indoor Localization](#indoor-localization)
  - [Challenges for Real-World Large-Scale WiFi Sensing](#challenges-for-real-world-large-scale-wifi-sensing)
    - [Efficiency and Security](#efficiency-and-security)
    - [Cross-Environment WiFi Sensing](#cross-environment-wifi-sensing)
    - [Multi-modal Sensing (WiFi+CV/Radar)](#multi-modal-sensing-wificvradar)
- [Platforms](#platforms)
  - [CSI Tool](#csi-tool)
  - [IoT Systems](#iot-systems)
- [Datasets](#datasets)
- [Libraries](#libraries)
- [Book Chapter](#book-chapter)

# Benchmark
* [Deep Learning and Its Applications in WiFi CSI Sensing]() | [[Github]]() A comprehensive benchmarking for deep learning models in WiFi sensing.

<!-- *********************************************************************** -->

# Papers

Papers are ordered by theme and inside each theme by publication date (submission date for arXiv papers).

## Methods
WiFi CSI sensing methods have enabled many applications, which can be divided into three categories:
* **Learning-based methods** learn the mapping functions from CSI data to the corresponding labels by machine learning and deep learning.
* **Modeling-based methods** are based on physical theories like the Fresnel Zone model, or statistical models like the Rician fading model.
* **Hybrid methods** derive the strengths from learning-based and modeling-based methods.

## Surveys
* [WiFi Sensing with Channel State Information: A Survey](https://www.cs.wm.edu/~yma/files/WiFiSensing_YongsenMa_authorversion.pdf) ACM Computing Surveys (2019)
* [Device-Free WiFi Human Sensing: From Pattern-Based to Model-Based Approaches](https://ieeexplore.ieee.org/abstract/document/8067692) IEEE Communications Magazine (2017)
* [From RSSI to CSI: Indoor Localization via Channel Response](https://dl.acm.org/doi/abs/10.1145/2543581.2543592) ACM Computing Surveys (2013)

## Applications
### Occupancy Detection
* [Machine Learning empowered Occupancy Sensing for Smart Buildings](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/icml2019/3/paper.pdf) IEEE ICML Climate Change Workshop (2019)
* [FreeDetector: Device-Free Occupancy Detection with Commodity WiFi](https://ieeexplore.ieee.org/abstract/document/8011040) IEEE International Conference on Sensing, Communication and Networking (SECON Workshops) (2017)

### Human Activity Recognition
* [Wi-Fi-Based Location-Independent Human Activity Recognition with Attention Mechanism Enhanced Method](https://www.mdpi.com/2079-9292/11/4/642/pdf) Electronics (2022)
* [Multimodal CSI-based Human Activity Recognition using GANs](https://ieeexplore.ieee.org/document/9431203) IEEE Internet of Things Journal (2021)
* [Improving WiFi-based Human Activity Recognition with Adaptive Initial State via One-shot Learning](https://ieeexplore.ieee.org/abstract/document/9417590) IEEE Wireless Communications and Networking Conference (2021)
* [Robust CSI-based Human Activity Recognition using Roaming Generator](https://ieeexplore.ieee.org/abstract/document/9305332) IEEE International Conference on Control and Automation (ICCA) (2020)
* [Device-free Occupancy Sensing Platform using WiFi-enabled IoT Devices for Smart Homes](https://ieeexplore.ieee.org/abstract/document/8391737) IEEE Internet of Things Journal (2018)
* [DeepSense: Device-Free Human Activity Recognition via Autoencoder Long-Term Recurrent Convolutional Network](https://ieeexplore.ieee.org/abstract/document/8422895) IEEE International Conference on Communications (ICC) (2018)
* [CareFi: Sedentary Behavior Monitoring System via Commodity WiFi Infrastructures](https://ieeexplore.ieee.org/document/8354831) IEEE Transactions on Vehicular Technology (2018)
* [Towards Occupant Activity Driven Smart Buildings via WiFi-enabled IoT Devices and Deep Learning](https://www.sciencedirect.com/science/article/abs/pii/S037877881831329X) Energy and Building (2018)
* [Poster:WiFi-based Device-free Human Activity Recognition via Automatic Representation Learning](https://www.researchgate.net/publication/320222015_Poster_WiFi-based_Device-Free_Human_Activity_Recognition_via_Automatic_Representation_Learning) Annual International Conference on Mobile Computing, MOBICOM-17 (2017)

### Human Identification

* [CAUTION: A Robust WiFi-based Human Authentication System via Few-shot Open-set Gait Recognition](https://ieeexplore.ieee.org/abstract/document/9726794/) IEEE Internet of Things Journal (2022)
* [Wifi-based Human Identification via Convex Tensor Shapelet Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11497) AAAI Conference on Artificial Intelligence AAAI-18 (2018)
 
### Crowd Counting
* [Device-free Occupancy Detection and Crowd Counting in Smart Buildings with WiFi-enabled IoT](https://www.sciencedirect.com/science/article/abs/pii/S0378778817339336) Energy and Building (2018)
* [FreeCount: Device-Free Crowd Counting with Commodity WiFi](https://ieeexplore.ieee.org/abstract/document/8255034) IEEE Global Communications Conference (2017)

### Gesture Recognition
* [WiHF: Enable User Identified Gesture Recognition with WiFi](https://ieeexplore.ieee.org/abstract/document/9155539) IEEE Conference on Computer Communications (INFOCOM) (2020)
* [Learning Gestures from WiFi: A Siamese Recurrent Convolutional Architecture](https://ieeexplore.ieee.org/document/8839094) IEEE Internet of Things Journal (2019)
* [Robust WiFi-Enabled Device-Free Gesture Recognition via Unsupervised Adversarial Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8487345/) | [[Github]](https://github.com/Marsrocky/WiADG) IEEE International Conference on Computer Communication and Networks (ICCCN) (2018)
* [WiFi-enabled Device-free Gesture Recognition for Smart Home Automation](https://ieeexplore.ieee.org/abstract/document/8444331) IEEE International Conference on Control and Automation (ICCA) (2018)
* [SignFi: Sign Language Recognition Using WiFi](https://dl.acm.org/doi/10.1145/3191755) | [[Github]](https://github.com/yongsen/SignFi) ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (2018)

### Fall Detection (for Healthcare)
- [A WiFi-Based Smart Home Fall Detection System Using Recurrent Neural Network](https://ieeexplore.ieee.org/abstract/document/9186064) IEEE Transactions on Consumer Electronics (2020)
* [RT-Fall: A Real-Time and Contactless Fall Detection System with Commodity WiFi Devices](https://ieeexplore.ieee.org/abstract/document/7458198) IEEE Transactions on Mobile Computing (2017)
* [FallDeFi: Ubiquitous Fall Detection using Commodity Wi-Fi Devices](https://dl.acm.org/doi/abs/10.1145/3161183) the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (2017)

### Respiration Detection
* [Device-Free Multi-Person Respiration Monitoring Using WiFi](https://ieeexplore.ieee.org/abstract/document/9179993) IEEE Transactions on Vehicular Technology (2020)
* [Contactless Respiration Monitoring Via Off-the-Shelf WiFi Devices](https://ieeexplore.ieee.org/abstract/document/7345587) IEEE Transactions on Mobile Computing (2016)
* [Human respiration detection with commodity wifi devices: do user location and body orientation matter?](https://dl.acm.org/doi/abs/10.1145/2971648.2971744) Ubicomp (2016)

### Indoor Localization
* [Passive Indoor Localization Based on CSI and Naive Bayes Classification](https://ieeexplore.ieee.org/abstract/document/7902212) IEEE Transactions on Systems, Man, and Cybernetics: Systems (2018)
* [CSI-Based Fingerprinting for Indoor Localization: A Deep Learning Approach](https://ieeexplore.ieee.org/abstract/document/7438932) IEEE Transactions on Vehicular Technology (2017)
* [CSI-Based Indoor Localization](https://ieeexplore.ieee.org/abstract/document/6244790) IEEE Transactions on Parallel and Distributed Systems (2013)

## Challenges for Real-World Large-Scale WiFi Sensing

### Efficiency and Security
* [EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression](https://ieeexplore.ieee.org/abstract/document/9667414) | [[Github]](https://github.com/Marsrocky/EfficientFi) IEEE Internet of Things Journal (2022) 
* [AutoFi: Towards Automatic WiFi Human Sensing via Geometric Self-Supervised Learning](https://arxiv.org/abs/2205.01629) arXiv:2205.01629 (2022)
* [RobustSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition](https://arxiv.org/abs/2204.01560) arXiv:2204.01560 (2022)
* [An Experimental Study of CSI Management to Preserve Location Privacy](https://dl.acm.org/doi/10.1145/3411276.3412187) | [[Github]](https://github.com/seemoo-lab/csicloak) ACM International Workshop on Wireless Network Testbeds, Experimental evaluation & Characterization (WiNTECH) (2020)

### Cross-Environment WiFi Sensing
* [Consensus Adversarial Domain Adaptation](https://ojs.aaai.org/index.php/AAAI/article/download/4552/4430) AAAI-19 (2019)
* [Joint Adversarial Domain Adaptation for Resilient WiFi-Enabled Device-Free Gesture Recognition](https://ieeexplore.ieee.org/abstract/document/8614062) IEEE International Conference on Machine Learning and Applications (2018)
* [Fine-Grained Adaptive Location-Independent Activity Recognition using Commodity WiFi](https://ieeexplore.ieee.org/abstract/document/8377133) IEEE Wireless Communications and Networking Conference (WCNC) (2018)

### Multi-modal Sensing (WiFi+CV/Radar)
* [WiFi and Vision Multimodal Learning for Accurate and Robust Device-Free Human Activity Recognition](https://openaccess.thecvf.com/content_CVPRW_2019/papers/MULA/Zou_WiFi_and_Vision_Multimodal_Learning_for_Accurate_and_Robust_Device-Free_CVPRW_2019_paper.pdf) IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2019)


<!-- *********************************************************************** -->

# Platforms

## CSI Tool
* [[Intel 5300 NIC]](https://dhalperi.github.io/linux-80211n-csitool/) Pioneered CSI tool enables Intel 5300 NIC to extract CSI data. It supports 30 subcarriers for a pair of antennas running on 20MHz.
* [[Atheros CSI Tool]](https://wands.sg/research/wifi/AtherosCSI/) Revamped CSI tool enables various Qualcomm Atheros NIC to extract CSI data. It supports 114 subcarriers for a pair of antennas running on 40MHz.
* [[Nexmon CSI Tool]](https://github.com/seemoo-lab/nexmon_csi) Mobile CSI Tool enables mobile phone and embedded device (RasPi) to extract CSI data of up to 256 subcarriers for a pair of antennas running on 80MHz.
* [[Software Defined Radio (SDR)]](https://www.ettus.com/) platforms, such as [Universal Software Radio Peripheral (USRP)](https://www.ettus.com/) and [Wireless Open Access Research Platform (WARP)](https://warpproject.org/trac), provide CSI measurements at 2.4GHz, 5GHz, and 60GHz.

## IoT Systems
* [Device-free occupant activity sensing using WiFi-enabled IoT devices for smart homes](https://ieeexplore.ieee.org/abstract/document/8391737/) Complete IoT-enabled WiFi system design for smart home.

# Datasets
* [[NTU-Fi]]() The NTU-Fi dataset is the only CSI dataset with 114 subcarriers per pair of antennas, collected using Atheros CSI tool. It consists of 6 human activities and 14 human gait patterns.
* [[Widar 3.0]](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset) The Widar 3.0 project is a large dataset designed for use in WiFi-based hand gesture recognition, collected using Intel 5300 NIC (30 subcarriers). The dataset consists of 258K instances of hand gestures with a duration of totally 8,620 minutes and from 75 domains.
* [[UT-HAR]](https://github.com/ermongroup/Wifi_Activity_Recognition) The dataset is collected in ''A Survey on Behaviour Recognition Using WiFi Channel State Information''. It consists of continuous CSI data for 6 activities without golden segmentation timestamp for each sample.
* [[SignFi]](https://github.com/yongsen/SignFi) Channel State Information (CSI) traces for sign language recognition using WiFi.

<!-- *********************************************************************** -->

# Libraries

* [[SenseFi]]() Deep learning libraries for WiFi CSI sensing (PyTorch)

# Book Chapter
* [Deep Learning and Unsupervised Domain Adaptation for WiFi-based Sensing](https://www.worldscientific.com/doi/abs/10.1142/9789811218842_0004) "Generalization with Deep Learning", World Scientific
