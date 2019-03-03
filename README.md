# NeuralTuringMachine
Tensorflow implementation of a Neural Turing Machine (simple version)

## 描述
基于 `https://github.com/MarkPKCollier/NeuralTuringMachine` 和 `https://arxiv.org/abs/1807.08518` 实现的神经图灵机简易版本。
只在copy任务中进行训练

## 实现
神经图灵机实现一个 `Controller`，基于 `LSTMCell`，在实现中继承 `tf.contrib.rnn.RNNCell` 实现神经图灵机的各种操作。
实现之后，可以使用 `tf.dynamic_rnn` 直接调用。这种实现方式更为简洁。
```
cell = NTMCell(controller_layers=1, controller_units=100, 
            memory_size=128, memory_vector_dim=20,              # 存储单元
            read_head_num=1, write_head_num=1,                  # 读写头数目
            addressing_mode='content_and_location', 
            shift_range=1, reuse=False, output_dim=8, clip_value=20, 
            init_mode='constant')
            
# 使用TF.dynamic_rnn运行. 
# 输入(batch, seq_len, dim)  输出 (batch, seq_len, num_units)
output_sequence, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            dtype=tf.float32,
            initial_state=None)
```

## 细节
本项目旨在实现神经图灵机的核心组成，没有对各超参数、初始化方式进行对比。从训练过程来看，前期的损失波动较大。可能需要更加细致的参数调整。
![损失](/img/loss.jpg)
![准确率](/img/acc.jpg)

## 可视化
### Epoch=100:
![Epoch=100](/save/epoch_100.jpg)
### Epoch=300
![Epoch=300](/save/epoch_300.jpg)
### Epoch=500
![Epoch=500](/save/epoch_500.jpg)
### Epoch=850
![Epoch=800](/save/epoch_850.jpg)
### Epoch=950
![Epoch=950](/save/epoch_950.jpg)




