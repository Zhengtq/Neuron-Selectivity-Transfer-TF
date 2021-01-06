#!/usr/bin/env python
#coding: utf-8

def tf_mmd_poly_loss(x_student, x_teacher):

    s_shape = tf.shape(x_student)
    s_batch_size = s_shape[0]
    s_height, s_width = s_shape[1], s_shape[2]
    s_depth = x_student.shape[3].value

    x_student_norm = tf.nn.l2_normalize(x_student, axis=(1,2))
    x_student_transpose = tf.transpose(x_student_norm, perm=[0, 3, 1, 2])
    x_student_reshape = tf.reshape(x_student_transpose, [s_batch_size, s_depth, s_height * s_width])
    x_student_reshape_T = tf.transpose(x_student_reshape, perm=[0, 2, 1])
    x_student_matrix = tf.einsum('bcw, bwt->bct', x_student_reshape, x_student_reshape_T)


    x_student_matrix_square = tf.multiply(x_student_matrix, x_student_matrix)
    x_student_matrix_mean = tf.reduce_mean(x_student_matrix_square, axis = (1,2))


    t_shape = tf.shape(x_teacher)
    t_batch_size = t_shape[0]
    t_height, t_width = t_shape[1], t_shape[2]
    t_depth = x_teacher.shape[3].value

    x_teacher_norm = tf.nn.l2_normalize(x_teacher, axis=(1,2))
    x_teacher_transpose = tf.transpose(x_teacher_norm, perm=[0, 3, 1, 2])
    x_teacher_reshape = tf.reshape(x_teacher_transpose, [t_batch_size, t_depth, t_height * t_width])
    x_teacher_reshape_T = tf.transpose(x_teacher_reshape, perm=[0, 2, 1])

    x_st_matrix = tf.einsum('bcw, bwt->bct', x_student_reshape, x_teacher_reshape_T)
    x_st_matrix_square = tf.multiply(x_st_matrix, x_st_matrix)
    x_st_matrix_mean = tf.reduce_mean(x_st_matrix_square, axis = (1, 2))

    mmd_loss = tf.reduce_mean(x_student_matrix_mean - 2 * x_st_matrix_mean)

    return mmd_loss
