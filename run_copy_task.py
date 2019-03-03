#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
from utils import plot_heatmap
# 导入定义的 NTM cell
from ntm import NTMCell

# 超参数设置
class Args:
    def __init__(self):
        self.learning_rate = 1e-2
        self.max_grad_norm = 10
        self.num_train_steps = 1000
        self.batch_size = 128
        self.num_bits_per_vector = 8
        self.max_seq_len = 20
args = Args()

# 产生训练数据
class CopyTaskData:
    def generate_batches(self, num_batches, batch_size, bits_per_vector=8, 
                         max_seq_len=args.max_seq_len):
        # 一个函数，如果 snap_boolean(x) 中 x>0.5 则结果为1，否则结果为0
        snap_boolean = np.vectorize(lambda x: 1.0 if x > 0.5 else 0.0)
        # 
        batches = []
        for i in range(num_batches):
            # 序列长度
            seq_len = np.random.randint(low=1, high=max_seq_len+1)
            self.seq_len = seq_len
            pad_to_len = seq_len

            # 返回值 x.shape=(seq_len, bits_per_vector+1)
            def generate_sequence():
                return np.asarray([snap_boolean(np.append(np.random.rand(
                    bits_per_vector), 0)) for _ in range(seq_len)])

            # 产生训练数据 (batch_size, seq_len, bits)
            inputs = np.asarray([generate_sequence() for _ in range(batch_size)]).astype(np.float32)
            # 全1数据
            eos = np.ones([batch_size, 1, bits_per_vector + 1])
            # 全0数据
            output_inputs = np.zeros_like(inputs)
            # 连接，在 seq_len 的维度
            full_inputs = np.concatenate((inputs, eos, output_inputs), axis=1)
            # 三个元素，第一个元素为 seq_len, 第二个为 full_inputs, 第三个为 inputs
            # 第三个为 label， label是input的一部分
            batches.append((pad_to_len, full_inputs, inputs[:, :, :bits_per_vector]))
        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        # 统计 labels 与 ouputs 的不同元素占整个序列的比例
        # labels.shape=(batch, seq_len, 8)=output.shape
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        np.sum(labels == outputs)
        bit_errors = np.sum(np.abs(labels - outputs)) / (args.batch_size*self.seq_len*args.num_bits_per_vector)
        return bit_errors

data_generator = CopyTaskData()
class BuildModel(object):
    def __init__(self, max_seq_len, inputs, labels):
        self.max_seq_len = max_seq_len
        self.inputs = inputs
        self.labels = labels
        self._build_model()
        self._cal_loss()

    def _build_model(self):
        # 模型定义
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
        
        self.output_logits = output_sequence[:, self.max_seq_len+1:, :]
        self.output = tf.sigmoid(self.output_logits)   

    def _cal_loss(self):
        # 计算损失
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.labels, logits=self.output_logits)
        self.loss = tf.reduce_sum(cross_entropy)/args.batch_size
        # train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), args.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

max_seq_len_placeholder = tf.placeholder(tf.int32)
inputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector+1))
outputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector))

#
model = BuildModel(max_seq_len_placeholder, inputs_placeholder, outputs_placeholder)
initializer = tf.global_variables_initializer()

# 训练
sess = tf.Session()
sess.run(initializer)

for i in range(args.num_train_steps):
    # 产生一个 batch 的样本. seq_len在0-20键随机采样，label的维度=seq_len, 
    # input中含有干扰因素，因而其维度为 seq_len*2+1. 例如: 10, (32, 21, 9), (32, 10, 8)
    seq_len, inputs, labels = data_generator.generate_batches(
        1,
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        max_seq_len=args.max_seq_len,
    )[0]
    # 损失计算
    train_loss, _, outputs = sess.run([model.loss, model.train_op, model.output],
        feed_dict={inputs_placeholder: inputs,
                   outputs_placeholder: labels,
                   max_seq_len_placeholder: seq_len
        })
    # 准确率计算
    avg_errors_per_seq = data_generator.error_per_seq(labels, outputs, args.batch_size)
    print('Epoch: ({0}), Loss : {1}'.format(i, train_loss/seq_len), ', acc: {0}%'.format((1.-avg_errors_per_seq)*100))
    # save heatmap
    if i % 50 == 0:
        plot_heatmap(i, inputs, outputs)
