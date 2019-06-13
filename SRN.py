"""
	Based on the PointNet++ codebase 
    https://github.com/charlesq34/pointnet2
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util


def SRNBlock(input_u, input_v, scope, bn, is_training, bn_decay):
    batchsize, length, in_uchannels = input_u.get_shape().as_list()
    _, _, in_vchannels = input_v.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        
        with tf.variable_scope('gu') as scope:
            relation_u = []
            for i in range(length):
                i_to_all = []
                for j in range(length):
                    i_to_all.append(input_u[:,i,:] + input_u[:,j,:])
                i_to_all = tf.stack(i_to_all, axis=1)
                relation_u.append(i_to_all)
            relation_u = tf.stack(relation_u, axis=1)
            gu_output = tf_util.conv2d(relation_u, 2*in_uchannels, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv1', bn_decay=bn_decay,
                                        data_format='NHWC')
        
            

        with tf.variable_scope('gv') as scope:
            relation_v = []
            for i in range(length):
                i_to_all = []
                for j in range(length):
                    i_to_all.append(tf.concat([input_v[:,i,:], input_v[:,j,:]], axis = -1))
                i_to_all = tf.stack(i_to_all, axis = 1)
                relation_v.append(i_to_all)
            relation_v = tf.stack(relation_v, axis = 1)
            gv_output = tf_util.conv2d(relation_v, 2*in_uchannels, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv2', bn_decay=bn_decay,
                                        data_format='NHWC')
        fused_uv = tf.concat([gu_output, gv_output], axis = -1)
        _,_,_, ch = fused_uv.get_shape().as_list()

        with tf.variable_scope('h') as scope:
            h_out = tf_util.conv2d(fused_uv, in_uchannels, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv3', bn_decay=bn_decay,
                                        data_format='NHWC')


        
        unsqueeze_h = tf.reduce_mean(h_out, axis=2)
        output = tf_util.conv1d(unsqueeze_h, in_uchannels, 1, padding='VALID', 
            activation_fn=tf.nn.relu, scope='conv4')
        
        output =  tf.concat([input_u + output, input_v], axis = -1)
        return output 


