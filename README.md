# 项目详解
## get_data.py (获取数据)
目前支持 MNIST, Fashion MNIST, CIFAR 10 和 CIFAR 100 数据集. 可以在```get_data.py``下自行替换成自己需要的数据集.

传入数据的格式为:
```
data_loader = {"train": train_loader, "valid": test_loader}
```
## get_model (获取模型)
目前支持:
- resnet18
- resnet34
- resnet50
- resnet101
- resnet152
- alexnet
- squeezenet
- vgg11
- vgg13
- vgg16
- vgg19

替换模型的方法:
```
python main.py --model_name "模型名称"
```
例如, 使用 vgg 13:
```
python main.py --model_name vgg13
```
例如, 使用 resnet 152:
```
python main.py --model_name resnet152
```
## 参数详解
必填参数:
- model_name: 模型名称, 类型为 string
- num_classes: 输出类别数, 类型为 int (例如 MNIST 是 10 分类, CIFAR 100 是 100 分类)


重要参数:
- data_name: 数据名称, 类型为 string, 默认为 CIFAR10
- data_gray: 是否为灰度图, 类型为 boolean, 默认为 False
- num_epochs: 迭代次数, 类型为 int, 默认为 20
- batch_size: 一个批次的样本数目, 默认为 512


可选参数 (不建议修改):
- feature_exact: 是否冻层, 类型为 boolean, 默认为 False
- use_pretrained: 是否使用预训练权重, 类型为 boolean, 默认为 True
- pretrained_model_path: 预训练权重, 类型为 string, 默认为 pretrained_model/
- model_save_path: 模型保存路径, 类型为 string, 默认为 "checkpoint/"
- visualize: 模型可视化, 类型为 boolean, 默认为 True

# 使用说明
首先我们需要```cd```到文件路径, 例如:
```
cd C:\Users\Windows\Desktop\Project\transfer_learning-main
```
## 训练 MNIST
使用 vgg13 训练 MNIST 数据集:
```
python main.py --data_name MNIST --data_gray True --model_name vgg13 --num_classes 10 --batch_size 2048
```
## 训练 Fashion MNIST
使用 vgg19 训练 Fashion MNIST 数据集:
```
python main.py --data_name FashionMNIST --data_gray True --model_name vgg19 --num_classes 10 --batch_size 2048
```
## 训练 CIFAR 10
使用 resnet18 训练 CIFAR 10 数据集:
```
python main.py --data_name CIFAR10 --model_name resnet18 --num_classes 10 --batch_size 2048
```
## 训练 CIFAR 100
使用 resnet152 训练 CIFAR 10 数据集:
```
python main.py --data_name CIFAR100 --model_name resnet152 --num_classes 100 --batch_size 2048
```
## 训练自己的数据
```
python main.py --data_name other --model_name ? --num_classes ? --batch_size ? --epochs ?
```
