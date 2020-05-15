# ADVENT实验
记录ADVENT 的复现实验和重新训练结果和分析

## 加载预训练模型在ADVENT 目录下测试：
### Advent
rlaunch --gpu=2 --cpu=30 --memory=8000 --python3 test.py --cfg ./advent/scripts/configs/advent_pretrained.yml
按照github主页上的要求，测试结果

<div align=center>

![alt text](Pics/ADVENT/1.png)

</div>

### Minent 总体略差，但是个别类别也有很好的
rlaunch --gpu=2 --cpu=30 --memory=80000 – python3 test.py --cfg ./advent/scripts/configs/minent_pretrained.yml
<div align=center>

![alt text](Pics/ADVENT/2.png)

</div>

### Advent +minent 效果最好
rlaunch --gpu=1 --cpu=30 --memory=80000 – python3 test.py --cfg ./advent/scripts/configs/advent+minent_pretrained.yml
<div align=center>

![alt text](Pics/ADVENT/3.png)

</div>

## 在ADVENT 下重新训练并测试结果：
### epoch = 118，000，mIoU: 41.56 比加载预训练模型差了2.2%
<div align=center>

![alt text](Pics/ADVENT/4.png)

</div>

### epoch = 96000 和预训练模型差了0.7%。  mIoU: 43.06
    model dir: /data/Projects/ADVENT/experiments/snapshots/GTA2Cityscapes_DeepLabv2_AdvEnt/model_96000.pth



### 训练结果分析
下图为网络训练的loss变化：
<div align=center>

![alt text](Pics/ADVENT/5.png)

</div>

对抗损失随着epoch增加而增加，但是网络测试的结果表明随着epoch增加，网络在目标域上的表现逐渐变好又逐渐变差。分析和具体原因如下：

根据代码，loss_adv_trg_*  是基于目标域图片的分割预测结果（进一步被转换为自信息）没有被判别器判别成和源域输出相似的损失，这个损失大，说明基于目标域的分割网络还没有很好地提取出合适的特征使得输出自信息和源域相似/使得输出的熵图较平整。

具体的原因可能是：
1. 因为分割网络没有在目标域上进行训练，因此没有提取出有判别力的特征，从而输出的熵会比较大，那么就不会跟目标域匹配了
2. 这个损失回传到特征提取网络之前还需要经过判别器，因此，回传的效率不高导致网络不能很好地学习。因而这一部分的学习需要进一步加强。（可以考虑将这一部分的训练多进行几次来提升效果。）


下面的图为训练之后，网络的熵图，原图和预测图：
<div align=center>

![alt text](Pics/ADVENT/6.png)
![alt text](Pics/ADVENT/7.png)
![alt text](Pics/ADVENT/8.png)
![alt text](Pics/ADVENT/9.png)
![alt text](Pics/ADVENT/10.png)
![alt text](Pics/ADVENT/11.png)

</div>









