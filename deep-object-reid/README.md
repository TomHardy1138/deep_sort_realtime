### Deep Object Re-identification

Repo based on top of https://github.com/openvinotoolkit/deep-object-reid

#### Main differences
 1) Stable reid working (last commits in original repo breaks it);
 2) Usage of Arcface and other metric losses instead of simple Cosface;
 3) Normalizations have compatability with openvino 2019-2020 (2021 in original repo);
 4) Preprocessing within network;
 5) Usage of custom dataset.
 6) TensorRT compatability.

#### Installation
    pip install -r requirements.txt
    # torch==1.4.0, onnx==1.6.0 highly recommended
    python setup.py develop

#### Dataset view
    $PATH_TO_DATA
    ├── common
    │   ├──1
    │   │    ├── *.JPEG
    │   ├──...
    │   ├──n
    │   │    ├── *.JPEG

#### Training
    cd deep-object-reid
    python scripts/main.py --config-file cfgs/cfg.yaml --root $PATH_TO_DATA data.save_dir $OUTPUT_PATH

#### Onnx preparation
    python tools/convert_to_onnx.py --config-file cfgs/cfg.yaml --output-name $ONNX_NAME.onnx model.load_weights $WEIGHTS