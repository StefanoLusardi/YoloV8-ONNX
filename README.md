# YoloV8-ONNX
Welcome to YoloV8-ONNX!

This is a project to show how to perform inference on a YoloV8 ONNX model using C++.

In order to simplify dependencies management as well as to improve performance, OpenCV is not required.

The algorithms contained in [pre_processing.hpp](./src/pre_processing.hpp/) and [post_processing.hpp](./src/post_processing.hpp/) are written in plain C++.

The only dependency is [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h) which is used to load images from disk.

## Getting Started

This project uses git submodules so it is required to clone it using the *--recurse-submodules* flag in order to retrive the required submodules.

```bash
# Clone the repo
git clone https://github.com/StefanoLusardi/YoloV8-ONNX --recurse-submodules
```

In order to kickstart this project it is required to configure a local development environment using the provided [scripts](./scripts/).

The only pre-requisite is to have a recent (i.e. >= 3.8) Python installed on your machine.

The development environment uses python-virtualenv to create a local environment in which all the project dependencies are installed (exactly as virtual environment is supposed to work for any ordinary Python projects). The virtual environment will be created in the *.venv* folder in the root of the repo.

Third party C++ dependencies will be installed using Conan Package Manager in a local cache folder. The conan cache will be created in the *.conan* folder in the root of the repo.

These preliminary steps must be done only the first time the project is bootstrapped: afterwards it is only required to activate the virtual environment in order to work with the project itself.

## Create development environment
In order to setup a development environment it is sufficient to run the script *scripts/env/setup.<bat|sh>* depending on your operating system.

```bash
# Linux/MacOS
chmod +x scripts/env/setup.sh
scripts/env/setup.sh

# Windows
scripts\env\setup.bat
```

## Install Python packages
Run this command into the previously created python virtual environment in order to install the required packages to download, export and validate the Yolo model.

```bash
pip install -r models/requirements.txt
```

## Download, Export and Validate YoloV8 model
Run these commands to download a pre-trained Yolo model from the Ultralytics repository, export into the ONNX format and validate it performing an inference with random data.

```bash
python models/yolo_create.py
python models/yolo_infer.py
```

After these commands you should have *yolov8n.pt* and *yolov8n.onnx* binaries into the root of your repository. In case you want to retrieve other Yolo binaries (e.g. "s", "m", "xl", ...) it is enough to replace the model name into the *models/yolo_create.py* script.


## Start development environment
In order to start the development environment it is sufficient to activate the Python Virtual Environment just created the step above.

```bash
# Linux/MacOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate.bat
```

## Setup Build Environment (Windows Only)
When building from command line on Windows it is necessary to activate the Visual Studio Developer Command Prompt.
Depending on the version of Visual Studio compiler and on its install location it is required to run *vcvars64.bat* script the set the development environment properly.
*Note*: using Visual Studio IDE or the CMake extension for VSCode this step is already managed in the background, so no action is required.
Examples:

```bash
# Visual Studio 2022 - Build Tools
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Visual Studio 2019 - Enterprise
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
```

## Setup Dependencies
Install the project dependencies as specified in the conanfile.txt.
```bash
python scripts/conan/setup.py <Debug|Release> <COMPILER_NAME> <COMPILER_VERSION>

# examples:
python scripts/conan/setup.py Release clang 15
python scripts/conan/setup.py Debug visual_studio 17

# Visual Studio versions:
# VS 2019: 16
# VS 2022: 17
```

## Build
Run the following script to configure, build and install the project.
```bash
python scripts/cmake.py <Debug|Release> <COMPILER_NAME> <COMPILER_VERSION>

# examples:
python scripts/cmake.py Release clang 15
python scripts/cmake.py Debug visual_studio 17
```

## Run
Once the project has been installed you can run it using the following command.
```bash
install/yolov8
```

This is an example of the expected output:
```bash
Model: yolov8n.onnx
Image: "images/dog.png"

Input: 0
 - name: images
 - shape: 1x3x640x640
 - element type: 1
 - element count: 1228800

Output: 0
 - name: output0
 - shape: 1x84x8400
 - element type: 1
 - element count: 705600

Detection results:
Class: 16, Confidence: 0.836329
Class: 16, Confidence: 0.83732
Class: 16, Confidence: 0.860839
```
