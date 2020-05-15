# Domain-Adapatation-note

These are the domain adaptation related note, it includes the paper reading, experiment record, and some reflections 
:octocat: 

# Related Paper Reading 
Paper Name | Links | description | experiment record
------------ | ------------- | ------------ | ------------
Category anchor-guided unsupervised domain adaptation for semantic segmentation_2019 | NIPS; [Github](https://github.com/RogerZhangzz/CAG_UDA)	|基于源域和目标域相同类别的特征向量在特征空间中距离较近的假设，把源域的每个类别上计算类别的平均值当成是类别中心，并促使源域的同一类别特征向量和目标域的激活特征向量向类别中心靠拢。同时给激活的特征分配伪标签，促使分类边界也根据目标域的标签进行相应的调整。	| 
An Adversarial Perturbation Oriented Domain Adaptation Approach for Semantic Segmentation_2020 | AAAI | 通过攻击特征层的方法，促使源域和目标域的特征相同，同时对小类别这类难以区分的部分造成扰动，使得分割网络的分类网络对每个类别同等看待，减少类别bias，促使mIOU提升。	| 
A DIRT-T approach to unsupervised domain adaptation_2018 | ICLR;[Github](https://github.com/RuiShu/dirt-t) | 从仅仅使用特征对齐的劣势出发，作者通过数学推理和实验证明：增加了cluster assumption损失约束的特征对齐方法能进一步通过避免提取的特征靠近分割边界从而提高跨域的分类精度。其中cluster assumption 代表高维空间中同一类的特征会聚集成为一个cluster，基于这个假设我们需要分类器的分类边界不从cluster中间跨过 | 
Self-ensembling for visual domain adaptation_2018 | ICLR;[Github](https://github.com/Britefury/self-ensemble-visual-domain-adapt)	| 使用两个不同参数的网络（包括dropout，给图片加noise，通过放射变换等图片增强方式）后，目标域的图片应当有同样的预测结果（因为没有改变语义结构）|
Collaborative and adversarial network for unsupervised domain adaptation_2018| CVPR |作者为了同时学习domain informativefeature 和domain uninformative feature（使得仅仅使用一个分类器就能进行分类，即训练的分类器可以直接应用于目标域的预测，同时提取这个特征也有利于学习），作者采用了collaborative and adversarial network.。并在随后使用置信度高且难以判别的特征产生的目标域标签作为伪标签进一步提高目标域预测器的表现|
Image to Image Translation for Domain Adaptation_2018| CVPR |通过一系列损失函数希望得到域不变且有判别性的特征|
DCAN: Dual Channel-wise Alignment Networks for Unsupervised Scene Adaptation_2018| ECCV | 通过instance norm做image translation 和feature alignment|复现里面的风格迁移部分用于标签迁移
Domain Adaptation for Semantic Segmentation with Maximum Squares Loss_2019| ICCV;[Github](https://github.com/ZJULearning/MaxSquareLoss) | 通过研究ADVENT中的熵优化的损失函数，作者发现其梯度值会随着概率的增大而呈指数型地增大，导致网络倾向于学习概率较大的像素（也就是本身就比较确定的像素）；同时由于像素标签分布的不均匀，出现较多的类别也会导致有更多的IOU提升，因此作者在损失函数前加了一项分类别的损失均衡项，使得出现较少的类别也能均衡地学习。|在ADVENT基础上改进的分割跨域实验，目前正在复现中
Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation_2019	| ICCV;[Github](https://github.com/postBG/DTA.pytorch) |用两个分类器/特征提取器的思想，迫使分类器的边界离特征足够远，特征离分类器的边界足够远。|在VISDA17上的分类实验，代码很强
CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency_2019| CVPR |基于image translation 前后图片结构不变，从而加入了输出标签的consistency loss，其他的就是常见的image translation， label transfer等|
All about Structure: Adapting Structural Information across Domains for Boosting Semantic Segmentation_2019	| CVPR;[Github](https://github.com/a514514772/%20DISE-Domain-Invariant-Structure-Extraction) |两个网络提取域不变和域变特征，并进行特征的排列组合，用于image translation 和label transfer|
Not All Areas Are Equal: Transfer Learning for Semantic Segmentation via Hierarchical Region Selection_2019| CVPR |（目标域的数据和标签均不足场景），使用标签来判断源域（image-level）的哪一部分与目标域相同，并仅仅使用这一部分来更新和训练网络。|
Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation_2019 | CVPR;[Github](https://github.com/jihanyang/AFN) |在分类跨域问题中发现源域的特征的范数都比较大，因此利用损失函数迫使目标域的范数增大用于训练分割网络	-	这个是分类实验，而且源码中没什么内容
Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation_2019 | CVPR;[Github](https://github.com/RoyalVane/CLAN) |用两个分类器分别对源域和目标域的特征进行分割，并且通过判别器进行输出空间对齐，以及促使两个分类器产生相同的结果，从而目标域的特征不会在分类器决策边界。同时对输出特征的相似度做一个weigting map，希望对于相似的输出特征减小回传损失，使得输出特征更好地匹配	
Learning to Adapt Structured Output Space for Semantic Segmentation_2018 | CVPR;[Github](https://github.com/wasidennis/AdaptSegNet) |将源域和目标域不同层次的特征映射的标签做判别，希望（不同特征下的）输出的标签空间对齐	
Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation_2019| CVPR |更新了距离度量为Sliced Wasserstein Discrepancy，进一步提升Maximum Classifier Discrepancy for Unsupervised Domain Adaptation 的效果。	
Maximum Classifier Discrepancy for Unsupervised Domain Adaptation_2018| CVPR;[Github](https://github.com/mil-tokyo/MCD_DA) |用两个不同的分类器和对抗的思想，使得和源域匹配的目标域特征不落在分类器的判别边界上。	| 两个判别器的基础实验，TBC
DLOW: Domain Flow for Adaptation DLOW: Domain Flow for Adaptation and Generalizationand Generalization_2019| CVPR |使用基于image translation + 特征匹配的跨域方法时，在image translation部分，不想直接将图片从源域映射到目标域，而是通过一个非向量变量z来生成介于源域和目标域之间的中间域图像，将源域图像替换为从而使得后续特征匹配更加容易。|
Constructing Self-motivated Pyramid Curriculums for Cross-Domain Semantic Segmentation: A Non-Adversarial Approach_2019|  ICCV |使用在源域上训练的分割器在目标域上的分割结果估计出不同尺度下的伪目标域标签，并用这个估计的标签结果和分割器分割出的结果做交叉熵损失；同时目标域和源域的标签分布也进行匹配。	-	
CyCADA: Cycle-Consistent Adversarial Domain Adaptation_2018|  ICML |结合两种特征来做域匹配的问题，即原始像素层次上（利用cycle-consistent loss和semantic consistent loss的做源域图到目标域图的转换）和提取的特征层次上。从而匹配低层次的特征（纹理，色调等）和高层次的特征（轮廓，结构等）|
Unsupervised domain adaptation by backpropagation_2015|  ICML |在语义分割基础上，通过加入判别器判别特征来自源域还是目标域	
ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation_2019 | CVPR;[Github](https://github.com/valeoai/ADVENT) |使用判别器使得目标域图像输出的预测结果的熵分布（文章中通过拆分成向量的形式，描述成加权自信息）和源域输出的预测结果的熵分布接近	 | 基于GTA5&Cityscape的分割实验
Conditional generative adversarial network for structured domain adaptaion_2018| CVPR |利用生成器生成额外的特征，从而从源域匹配到目标域|
Domain Randomization and Pyramid Consistency：simulation--to-real generalization without accessing target domain data_2019|  ICCV |将源域数据集通过其他辅助数据集生成多种风格迁移效果的数据集，并将这个数据应用于分割网络

# Experiment Record	
Following is the dir tree 

