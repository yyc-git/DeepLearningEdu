卷积层的梯度检测
    


<!-- 重构代码，分离层 -->
重构代码，加入调试日志
    加上测试代码（同学不用实现测试代码，只需要知道怎么用就行）


实现LeNet识别手写数字
    （未收敛）

使用He Normal初始化


    实现LeNet识别手写数字
        （已收敛）



使用Adam

    实现LeNet识别手写数字
        （不需要He Normal，用Adam替代，也会收敛）







# finish 基础课









# publish AI3D



AI3D:

extract graph mr
    add graph check debug
        e.g. check whether layer's input and output count are equal or not


加入更多后端:
train:
CPU(implement by own)
WebGPU(
    ////encapulate Tensorflow.js? 
or implement by own?
)

inference:
CPU(implement by own)
WebNN




Extend:
extend by add layer(backend: CPU, WebGPU, WebNN)
extend by add activate




Meta3D:
use Meta3D
    Meta3d should finish 0.9.0 first!!!






LeNet: train by WebGPU, inference by WebNN





Publish AI3D






# begin 提高课




Dropout层的前向和后向传播

L2范式

实现AlexNet






实现VGG


实现NiN



实现GoogleLeNet




BN层的前向和后向传播

实现残差网络ResNet






<!-- 实现DenseNet -->





使用ResNet识别图片类别
















开始降噪
DnCNN


使用DnCNN降噪






<!-- 继续降噪 -->
<!-- 改进DnCNN -->



refer to: 
https://zhuanlan.zhihu.com/p/410980303
https://zhuanlan.zhihu.com/p/421052050



LBF
A machine learning approach for filtering Monte Carlo noise




KPCN
Kernel-predicting convolutional networks for denoising Monte Carlo renderings




KPAL???





Monte Carlo Denoising via Auxiliary Feature Guided Self-Attention





Self-Supervised Post-Correction for Monte Carlo Denoising
https://github.com/CGLab-GIST/self-supervised-post-corr





