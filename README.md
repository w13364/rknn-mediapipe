模型平台：rk3568（必须要转成自己的开发板版本）
```
环境配置：https://github.com/airockchip/rknn-toolkit2
rknn-toolkit2
python3.8
rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_aarch64
manylinux2014_aarch64.whl,arm64_requirements_cp38.txt


把这三个文件拷贝到对应你的include和lib里，工程里也有。
如果觉得安装包太大，只装rknn-toolkit-lite2-v2.0.0b1对应的python版本，然后拷贝下面三个文件到对应目录。
```


```
rknn-toolkit2-master
│      
└── rknpu2
    │      
    └── runtime
        │       
        └── Linux
            │      
            └── librknn_api
                ├── aarch64
                │   └── librknnrt.so
                └── include
                    ├── rknn_api.h
                    ├── rknn_custom_op.h
                    └── rknn_matmul_api.h

$ cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/aarch64
$ sudo cp ./librknnrt.so /usr/local/lib
$ cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/include
$ sudo cp ./rknn_* /usr/local/include
```


```
运行指令



根目录下：

make -f Makefile.blaze_rknn clean && make -f Makefile.blaze_rknn

./blaze_detect_single_rknn 8.jpg　（单张图推理）
./blaze_detect_video_rknn output.mp4　（视频流推理）
./blaze_detect_live_rknn （摄像头流推理）
```


```
Demo情况，推理单图耗时：121ms

![8365d5dd1414cb00b5e89223db33dded](https://github.com/user-attachments/assets/e493164a-c060-4073-a14a-a30481d67dbe)
```



