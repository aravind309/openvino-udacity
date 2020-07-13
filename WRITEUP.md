# People Counter application

This project demonstrates the use of people counter app across different Intel hardware like CPU,GPU,MYRIAD,HDDL-R and FPGA. The main focus in on converting model and running inference on difference accelerators.

This is tested on Ubuntu 16 and Ubuntu 18 with FP32 and FP16 precesions on CPU and iGPU. Hardware device used is upsquared  core board.

Prerequisites : Please refer to dependencies of nodejs and npm requirements from README.md. This current repo is tested on both Ubuntu 18 and Ubuntu 16 with Intel® Distribution of OpenVINO™ toolkit 2020.2. For installation refer to[Open Vino installation](https://docs.openvinotoolkit.org/2020.2/_docs_install_guides_installing_openvino_linux.html). 

## Explaining Custom Layers

The Intel® Distribution of OpenVINO™ toolkit supports neural network model layers in multiple frameworks including TensorFlow*, Caffe*, MXNet*, Kaldi* and ONYX*. The list of known layers is different for each of the supported frameworks. To see the layers supported by your framework, refer to supported frameworks[Supported Framework layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)
In Latest versions of OpenVINO CPU extension is in built. After querying the network if we have any unsupported layers we can add them to Model optimizer. Please refer to [Custom Layers](https://docs.openvinotoolkit.org/2020.2/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html) for implementation.

#### Model optimizer

The Model Optimizer of OpenVINO helps us to convert models(.pb files, .onnx files etc) in multiple different frameworks to IR files which will be used Inference Engine of OpenVINO. The main adavntage of using Model optimizer is it will optimize the models by shriking model size and accelerates the speed.
The model optimizer supports at present INT8, FP32, FP16 outputs. The is always a compramise between model size and speed. If model size is large(higher precision) speed willbe lower but the accuracy will be higher.


Below are the various model optimization techinques used by OpenVINO.

##### Techniques
 - Linear Operations Fusing
 - Stride Optimization
 - Disable Fusing
 
 The other main advantage of Model optimizer is we can cut the models and part of the models can be removed.
 
 Detailed documentation of Model optimizer can be found at [MO documentation](https://docs.openvinotoolkit.org/2020.2/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

 #### Inference Engine

Inference Engine  will run the actual inference of model. It only works with the IR that come from the Model Optimizer, or pretrained models which are in IR format. In geenral it will have .xml, .bin(weights) and a mapping file.

Inference Engine also helps in optimization such as  hardware-based optimizations to get even further improvements from a model. This helps our application  application to run at the edge and use up as little of device resources as possible.
These APIS are a unified API to allow high performance inference on many hardware types including Intel CPU, Intel iGPU, Intel FPGA, Intel Movidius Neural Compute Stick(HDDL-R), and Intel NCS stick. It also has API's to get the to query device information.

 Detailed documentation of Model optimizer can be found at [IE documentation](https://docs.openvinotoolkit.org/2020.2/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)




## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

 - Self Checkout system in retail marts : Retail mart can guide the customers to go to a vacant checkout systems and help in better management of systems. It can even turn out the checkout systems if it identifies less number of people which is environmental firendly
 - Identify number of people in the scene: It helps in identifying the number of people/audience in the frame and trigger an alarm if it crosses threshold. This system with some modification can be leveraged in COVID-19 pandemic scenarios to identify the people with/without masks, distance between two persons, number of individuals in a given system
 - Smart cities advanced transport coordination: Identify number of people in public bus stops and transit centers and send special buses to avoid crowds.
 - Identify hotspots in public place: Idnetify unusal hotspots which occur during accident/crime scenarios and alert the police and respective authorities
 - Identify traffic violation: It can be used to identify the number of persons travelling on a bike/auto



## Model Research

In investigating potential people counter models, I tried below data models:

- Model 1: SSD Mobilenet
  - [Model Source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  - Convert a model Intermediate Representation with below arguments
  
```
python3 /opt/intel/openvino_2020.2.120/deployment_tools/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --data_type FP32 --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```

  - This model is not having good accuracy.

  
- Model 2:Tiny Yolo V2]
  - [Model Source](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx)
  - Convert a model Intermediate Representation with below arguments
  
```
python3 /opt/intel/openvino_2020.2.120/deployment_tools/model_optimizer/mo.py --input_shape [1,3,416,416] --input_model tinyyolov2.onnx --data_type   FP32 
```
  
  - The model is little heavy and it is taking around 140 ms whereas predefined model is around ~30 ms. In a tradeoff I decided to use a model with higher inference rate.


- Model 3: [Mask RCNN]
  - [Model Source](https://www.cis.upenn.edu/~jshi/ped_html/)
  - Convert a model Intermediate Representation with below arguments
  
```
cd /opt/intel/openvino_2020.2.120/deployment_tools/model_optimizer

python3 ./mo_onnx.py
--input_model mask_rcnn_x.onnx \
--input "0:2" \
--input_shape [1,3,800,800] \
--mean_values [102.9801,115.9465,122.7717] \
--transformations_config ./extensions/front/onnx/mask_rcnn.json 
```
  - The model is heavy and i could see lot of resouces of my computer are consumed and couldn't load it on iGPU. While this a model in pytorch based using converson I convereted to onnx files. I could see infact the inference rate is very less(0.5 FPS). I also need to do a lot of Transfer Learning stuff to get it to stable ONNX format file.

## The Model

After exploring lot of models I could see pretrained models are doing a better job and good speed 
 
- [person-detection-retail-0002](https://docs.openvinotoolkit.org/2020.2/person-detection-retail-0002.html)
- [person-detection-retail-0013](https://docs.openvinotoolkit.org/2020.2/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)



### Downloading model

Download all the pre-requisite libraries and source the openvino installation using the following commands:

```sh
source /opt/intel/openvino/bin/setupvars.sh
```

Navigate to the directory containing the Model Downloader:

```sh
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
```


```sh
sudo ./downloader.py --name person-detection-retail-0013 --precisions FP32 -o /home/workspace

sudo ./downloader.py --name person-detection-retail-0002 --precisions FP32 -o /home/workspace
```
Please change the precesion to FP16 and explore more if required.
You can skip the precisions paramter for which the script downloads all precesions.

## Comparing Model Performance

As discussed above I was maily looking at inference rate and ended up using the pre ttrained models which have a added advantage of accuracy, size and time

### Inference time on CPU

| |person-detection-retail-002|person-detection-retail-0013
|-|-|-|-|
|FP32|43 ms|38 ms|
|FP16|32 ms|31 ms|


Finaly I chossed person-detection-retail-0013 with FP32 precession which is good tradeoff between accuracy and inference time.

## Performing Inference on local edge system.

Please follow the instruction in Readme.md if you are working in workspace.
In case if you deploying on real edge device and to avoid overhead of browser and other softwares which will alter the inference rate.

Open a new terminal. Clone the repo and go into the repo.

Execute the below commands:

```sh
  cd webservice/server
  npm install
```
After installation, run:

```sh
  cd node-server
  node ./server.js
```

If succesful you should receive a message-

```sh
Mosca Server Started.
```

```

Finally execute the below command in another terminal

This peice of code specifies the testing video provided in `resources/` folder and run it on port `3004`

```sh
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013/FP32/person-detection-retail-0013.xml  -d CPU -pt 0.6 
```
