#SiamRPN,SiamRPNPP 的重构

本项目参考自pysot，但好处是打了一些注释。

数据集代码包含TC，DTB70，UAV123，Coco

我记得这些数据集有一些需要数据先处理一下，比如Coco的gt有一些就很小，无法使用，之后出一个数据集处理的项目吧。。。

现在包含了两个算法SiamRPN和SiamRPNPP，区别就是一个骨干网络用Alexnet另一个用Restnet。。。

SiamRPNPP的模型超过github上传的100m要求了，我就全搞百度云盘了 https://pan.baidu.com/s/1gacTF5h5nQwTwQWlxgWD8A?pwd=2333 提取码: 2333 

根据我的训练经验，用Coco训练效果会比较好，跟踪数据集的样本还是太少了，就那几个目标。

里边也许有bug，因为这个是我从现在毕设的项目中摘出来的一部分，毕设暂时保密，等我毕了业再说，233333

