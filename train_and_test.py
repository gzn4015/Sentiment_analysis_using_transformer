import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from Module_Structure import Transformer, TokenEmbedding, PositionalEncoding, Encoder, ClassifierHead


def Train_And_Test(config, train_dataloader, val_dataloader, test_dataloader, num=20):    # 函数功能: 实现模型的训练以及测试
    # 函数参数：config:模型参数字典对象
    # train_dataloader, val_dataloader, test_dataloader分别为训练、验证、测试数据的DataLoader对象（数据加载器）
    # num是在每轮训练中经过多少批次的训练后（默认是经过20批次的训练），进行一次验证
    # Instantiate a transformer model
    model = Transformer(config).to(config.device)  # 实例化transformer模型，并将参数字典传递给transformer
    # .to(config.device)将模型移动到gpu或者cpu上

    # Define loss function and optimizer
    loss_function = torch.nn.BCELoss()  # 定义损失函数为二元交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # 定义优化器为adam，用来调整模型参数，并设置学习率为1e-6

    # Move model to GPU if available
    model = model.to(config.device)  # 如果 GPU 可用，将模型移动到 GPU 上进行训练，以提高训练速度

    # Number of training epochs
    n_epochs = 2  # 设置训练轮次为8

    # Metrics dictionary for plotting later
    metrics = {  # 指标字典，用于存储训练过程中的损失和准确率等指标，以便后续绘制训练曲线。
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    # 模型训练和验证
    for epoch in range(n_epochs):  # 在每个epoch里迭代训练
        # Create a tqdm progress bar for the training data
        train_data_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}', leave=False)
        # 使用 tqdm 库创建一个进度条，显示训练数据的迭代进度，train_dataloader是先前创建的数据加载器，用于从训练集加载数据
        for i, (inputs, targets) in enumerate(train_data_iterator):  # 对训练数据进行迭代，inputs 是模型的输入数据，targets 是对应的目标标签
            # Move inputs and targets to device
            inputs = inputs.to(config.device)  # 将输入数据和对应标签移动到指定设备上
            targets = targets.to(config.device)

            # Set model to training mode
            model.train()  # 设置模型为训练模式，这会启用 Dropout 和 Batch Normalization 等层的训练行为

            # Clear the gradients
            optimizer.zero_grad()  # 清除先前迭代中累积的梯度

            # Forward pass
            outputs = model(inputs)  # 对输入数据进行前向传播，得到输出数据

            # Compute loss
            train_loss = loss_function(outputs.squeeze(), targets)  # 使用损失函数计算模型的输出与实际标签的损失

            # Backward pass and optimize
            train_loss.backward()  # 执行反向传播
            optimizer.step()  # 更新模型参数

            # Calculate accuracy
            train_predictions = (outputs.squeeze() > 0.5).float()  # 二值化输出
            # outputs.squeeze() > 0.5 将模型输出的概率值转换为二进制标签（0 或 1）。大于 0.5 的值被视为 1，其他则为 0，并将其强转为float类型
            train_accuracy = (train_predictions == targets).float().mean().item()  # 计算训练数据的准确率

            # Update tqdm progress bar description with loss and accuracy
            train_data_iterator.set_postfix(
                {'Train Loss': train_loss.item(), 'Train Accuracy': train_accuracy})  # 更新进度条

            # Validation loop
            if (i % num==0) and (i != 0) == 1:  # 每循环上述步骤num次，就进行验证，从而减少验证的频率，以提高训练速度
                model.eval()  # 将模型设置为评估模式（验证和测试时需设置为评估模式）
                with torch.no_grad():  # 设置包含在其内部的代码禁用梯度计算（一般在推理和验证阶段设置）
                    val_losses = []  # 定义两个空列表来存放验证数据集上的损失以及准确率
                    val_accuracies = []

                    for val_inputs, val_targets in val_dataloader:  # 循环验证数据加载器里的验证数据以及其对应的标签
                        val_inputs = val_inputs.to(config.device)  # 将验证数据和其对应的标签移动到指定设备上
                        val_targets = val_targets.to(config.device)

                        val_outputs = model(val_inputs)  # 前向传播，获得模型的输出
                        val_loss = loss_function(val_outputs.squeeze(), val_targets)  # 根据模型输出和实际标签，计算二者间的损失
                        val_losses.append(val_loss.item())  # 将当前批次的验证损失添加到 val_losses 列表中。val_loss.item() 获取损失值的标量表示

                        val_predictions = (val_outputs.squeeze() > 0.5).float()  # 二值化输出
                        # val_outputs.squeeze() > 0.5 将模型输出的概率值转换为二进制标签（0 或 1）。大于 0.5 的值被视为 1，其他则为 0，并将其强转为float类型
                        val_accuracy = (val_predictions == val_targets).float().mean().item()  # 计算验证数据集的准确率
                        val_accuracies.append(val_accuracy)  # 将本次验证的准确率添加到列表val_accuracies里

                # Get the mean loss and accuracy over the validation set
                val_loss = np.mean(val_losses)  # 计算验证集的平均损失和准确率
                val_accuracy = np.mean(val_accuracies)

                # Print metrics during training
                print(
                    f"Epoch {epoch + 1}/{n_epochs} Step {i} \tTrain Loss: {train_loss.item():.2f} \tTrain Accuracy: {train_accuracy:.3f}\n\t\t\tVal Loss: {val_loss:.2f}   \tVal Accuracy: {val_accuracy:.3f}")
                # 输出显示训练和验证集上的指标
                # Store metrics
                metrics['train_loss'].append(train_loss.item())  # 将训练和验证数据的指标存放进对应的指标列表里
                metrics['train_accuracy'].append(train_accuracy)
                metrics['val_loss'].append(val_loss)
                metrics['val_accuracy'].append(val_accuracy)
    torch.save(model.state_dict(), '../Sentiment analysis using transformer/module.pth')   # 保存训练完的模型参数字典

    # 模型测试
    model.eval()  # 设置模型为评估模式（验证和测试时需设置为评估模式）
    test_losses = []  # 定义连哥哥列表分别存放测试损失和准确率
    test_accuracies = []

    with torch.no_grad():  # 设置包含在其内部的代码块禁用梯度计算
        for test_inputs, test_targets in tqdm(test_dataloader, desc="Testing"):  # 使用 tqdm 库创建一个进度条来显示测试数据的迭代进度
            # test_dataloader 是测试数据加载器，test_inputs 是输入数据，test_targets 是对应的目标标签
            test_inputs = test_inputs.to(config.device)  # 将输入数据和对应标签移动到指定设备上
            test_targets = test_targets.to(config.device)

            test_outputs = model(test_inputs)  # 前向传播，获得模型输出
            test_loss = loss_function(test_outputs.squeeze(), test_targets)  # 计算损失
            test_losses.append(test_loss.item())  # 将损失添加到损失列表里

            test_predictions = (test_outputs.squeeze() > 0.5).float()  # 二值化输出
            test_accuracy = (test_predictions == test_targets).float().mean().item()  # 计算准确率
            test_accuracies.append(test_accuracy)  # 将准确率添加到准确率列表里

    # Get the mean loss and accuracy over the test set
    mean_test_loss = np.mean(test_losses)  # 计算平均损失和准确率
    mean_test_accuracy = np.mean(test_accuracies)

    print(f"Test Loss: {mean_test_loss:.2f} \tTest Accuracy: {mean_test_accuracy:.3f}")  # 输出显示测试指标
