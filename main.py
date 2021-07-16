import os
import argparse
import time
import copy
import torch
from torchsummary import summary
from get_model import initialize_model
from get_data import get_data
from visualize import visualize_model


def train_model(model, dataloaders, data_gray, criterion, optimizer, model_save_path, num_epochs=25):
    # 获取起始时间
    since = time.time()

    # checkpoint路径
    checkpoint_name = time.strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(model_save_path + checkpoint_name)

    # 初始化参数
    best_acc = 0
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 训练和验证
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:

                # 转换成RGB
                if data_gray:
                    inputs = inputs.repeat(1, 3, 1, 1)

                # cuda 加速
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # 计算损失
                    loss = criterion(outputs, labels)

                    # 训练阶段更新权重
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_eplased = time.time() - since
            print("Time elapsed {:.0f}m {:.0f}s".format(time_eplased // 60, time_eplased % 60))
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # 得到最好的模型
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                state = {
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                }

                # 保存模型
                torch.save(state, model_save_path + checkpoint_name + "/checkpoint.pth")
            if phase == "valid":
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print("Optimizer learning rate: {:.7f}".format(optimizer.param_groups[0]["lr"]))
        LRs.append(optimizer.param_groups[0]["lr"])
        print()

    time_eplased = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_eplased // 60, time_eplased % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_weights)

    # 返回
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


def parse_opt():
    parser = argparse.ArgumentParser()

    # 必选参数
    parser.add_argument('--model_name', type=str, required=True, help="模型")
    parser.add_argument('--num_classes', type=int, required=True, help="输出类别数")

    # 重要参数
    parser.add_argument('--data_name', type=str, default="CIFAR10", help="数据名称 MNIST | FashionMNIST | CIFAR10 | CIFAR100 | other")
    parser.add_argument('--data_gray', type=bool, default=False, help="数据是否是单通道")
    parser.add_argument('--num_epochs', type=int, default=20, help="迭代次数")
    parser.add_argument('--batch_size', type=int, default=512, help="一次训练的样本数目")

    # 可选参数
    parser.add_argument('--feature_exact', type=bool, default=False, help="冻层, 默认为 False")
    parser.add_argument('--use_pretrained', type=bool, default=True, help="使用预训练模型")
    parser.add_argument('--pretrained_model_path', type=str, default="pretrained_model/", help="使用预训练模型")
    parser.add_argument('--model_save_path', type=str, default="checkpoint/", help="模型保存")
    parser.add_argument('--visualize', type=bool, default=True, help="模型可视化")

    args = parser.parse_args()

    return args


def params_initialize(model, params_to_update):
    # 是否使用GPU训练
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda: model.cuda()  # GPU 计算
    print("------------------------\n是否使用 GPU 加速:\n=======================\n", use_cuda)

    # 输出网络结构
    print(summary(model, (3, 32, 32)))

    optimizer = torch.optim.Adam(params_to_update, lr=0.01)  # 优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率每10个epoch衰减到原来的1/10
    criterion = torch.nn.NLLLoss()  # 损失函数

    return device, optimizer, scheduler, criterion


if __name__ == "__main__":

    # 初始化参数
    args = parse_opt()
    print(args)

    # 预训练模型路径
    os.environ['TORCH_HOME'] = args.pretrained_model_path

    # 获取数据
    data_loader, class_names = get_data(
        data_name=args.data_name,
        data_gray=args.data_gray,
        batch_size=args.batch_size
    )
    print("class names:", class_names)

    # 获取模型和需要更新的参数
    model, params_to_update = initialize_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        feature_exact=args.feature_exact,
        use_pretrained=args.use_pretrained
    )

    # 模型参数初始化
    device, optimizer, scheduler, criterion = params_initialize(model, params_to_update)

    # 开始训练
    model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
        model=model,
        dataloaders=data_loader,
        data_gray=args.data_gray,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        model_save_path=args.model_save_path
    )

    # 模型可视化
    visualize_model(
        visualize=args.visualize,
        model=model,
        dataloaders=data_loader,
        class_names=class_names,
        device=device
    )
