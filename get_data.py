import torchvision
from torch.utils.data import DataLoader


class Data_Name_Error(Exception, ):
    def __init__(self, message):
        super(Data_Name_Error, self).__init__(message)
        self.message = "data_name can only be: {}".format(message)


def get_data(data_name, data_gray, batch_size):
    """获取数据"""

    train_loader = None
    test_loader = None

    # 判断输入的模型名是否在支持的模型数组内
    try:
        data_name_array = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "other"]

        if data_name not in data_name_array:
            raise Data_Name_Error(data_name_array)

    except Data_Name_Error as e:
        raise (e)

    if data_gray:
        # 转换
        pre_process = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # 转换成张量
            torchvision.transforms.Normalize((0.5), (0.5))  # 标准化
        ])
    else:
        # 转换
        pre_process = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # 转换成张量
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
        ])

    if data_name == "MNIST":
        """MNIST"""

        # 获取测试集
        train = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=pre_process
        )

        train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

        # 获取测试集
        test = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=pre_process
        )

        test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    if data_name == "FashionMNIST":
        """FashionMNIST"""

        # 获取测试集
        train = torchvision.datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=pre_process
        )

        train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

        # 获取测试集
        test = torchvision.datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=pre_process
        )

        test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    if data_name == "CIFAR10":
        """CIFAR10"""

        # 获取测试集
        train = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=pre_process
        )

        train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

        # 获取测试集
        test = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=pre_process
        )

        test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    if data_name == "CIFAR100":
        """CIFAR100"""

        # 获取测试集
        train = torchvision.datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=pre_process
        )
        train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

        # 获取测试集
        test = torchvision.datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=pre_process
        )
        test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    if data_name == "other":
        """Write your own code here"""
        pass

    # 整合
    data_loader = {"train": train_loader, "valid": test_loader}

    # 返回分割好的训练集和测试集
    return data_loader, train.classes