import torch
import torchvision.models as models


def initialize_model(model_name, num_classes, feature_exact, use_pretrained=True):
    """
    初始化模型
    :param model_name: 模型名字
    :param num_classes: 类别数
    :param feature_exact: 是否冻层
    :param use_pretrained: 是否下载模型
    :return: 返回模型,
    """

    model_ft = None

    if model_name == "resnet18":
        """Resnet18"""

        # 加载模型
        model_ft = models.resnet18(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    if model_name == "resnet34":
        """ResNet34"""

        # 加载模型
        model_ft = models.resnet34(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    if model_name == "resnet50":
        """ResNet50"""

        # 加载模型
        model_ft = models.resnet50(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    if model_name == "resnet101":
        """ResNet101"""

        # 加载模型
        model_ft = models.resnet101(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    if model_name == "resnet152":
        """ResNet152"""

        # 加载模型
        model_ft = models.resnet152(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    if model_name == "alexnet":
        """AlexNet"""

        # 加载模型
        model_ft = models.alexnet(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_features, num_classes)

    if model_name == "vgg11":
        """Vgg11"""

        # 加载模型
        model_ft = models.vgg11(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_features, num_classes)

    if model_name == "vgg13":
        """Vgg13"""

        # 加载模型
        model_ft = models.vgg13(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_features, num_classes)

    if model_name == "vgg16":
        """Vgg16"""

        # 加载模型
        model_ft = models.vgg16(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_features, num_classes)

    if model_name == "vgg19":
        """Vgg19"""

        # 加载模型
        model_ft = models.vgg19(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_features, num_classes)

    if model_name == "squeezenet":
        """SqueezeNet"""

        # 加载模型
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)  # 下载参数
        set_parameter_requires_grad(model_ft, feature_exact)  # 冻层

        # 修改全连接层
        model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    # 获取需要更新的参数
    parameter_ft = parameter_to_update(model=model_ft, feature_exact=feature_exact)

    # 返回初始化好的模型
    return model_ft, parameter_ft


def set_parameter_requires_grad(model, feature_extracting):
    """
    是否保留梯度, 实现冻层
    :param model: 模型
    :param feature_extracting: 是否冻层
    :return: 无返回值
    """

    if feature_extracting:  # 如果冻层
        for param in model.parameters():  # 遍历每个权重参数
            param.requires_grad = False  # 保留梯度为False


def parameter_to_update(model, feature_exact):
    """
    获取需要更新的参数
    :param model: 模型
    :return: 需要更新的参数列表
    """

    print("Params to learn")
    param_array = model.parameters()

    if feature_exact:
        param_array = []
        for name, param, in model.named_parameters():
            if param.requires_grad == True:
                param_array.append(param)
                print("\t", name)
    else:
        for name, param, in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    return param_array
