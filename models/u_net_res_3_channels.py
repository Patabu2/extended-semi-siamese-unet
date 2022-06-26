import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, ReLU, Add


#---------------------------------------------------------
# Residual Block
#---------------------------------------------------------
class ResBlock(Model):
    def __init__(self, filters, padding, strides):
        super(ResBlock, self).__init__()

        self.conv_1 = Conv2D(filters, kernel_size = (3,3), padding = padding, strides = strides, activation = 'relu')
        self.conv_2 = Conv2D(filters, kernel_size = (1, 1), padding = padding, strides = 1, activation = 'relu')
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.relu = ReLU()
        self.addition = Add()

    def call(self, inputs):
        # First path
        x_1 = self.conv_1(inputs)
        x_1 = self.batch_norm_1(x_1)
        x_1 = self.relu(x_1)

        # Second path
        x_2 = self.conv_2(inputs)
        x_2 = self.batch_norm_2(x_2)
        x_2 = self.relu(x_2)

        # Join both paths
        output = self.addition([x_1, x_2])

        return output


#--------------------------------------------------------------------------------
# Super Crazy Dense Block :'v
#--------------------------------------------------------------------------------
class DenseBlock(Model):
    def __init__(self):
        super(DenseBlock, self).__init__()

        small_conv = 8
        big_conv = 32
        self.relu = ReLU()
        self.concat = Concatenate()

        self.conv_1 = Conv2D(big_conv, kernel_size = (1,1), padding = 'same', strides = 1)
        self.batch_n1 = BatchNormalization()
        
        self.conv_2 = Conv2D(small_conv, kernel_size = (3,3), padding = 'same', strides = 1)
        self.batch_n2 = BatchNormalization()

        self.conv_3 = Conv2D(big_conv, kernel_size = (1,1), padding = 'same', strides = 1)
        self.batch_n3 = BatchNormalization()

        self.conv_4 = Conv2D(small_conv, kernel_size = (3,3), padding = 'same', strides = 1)
        self.batch_n4 = BatchNormalization()

        self.conv_5 = Conv2D(big_conv, kernel_size = (1,1), padding = 'same', strides = 1)
        self.batch_n5 = BatchNormalization()

        self.conv_6 = Conv2D(small_conv, kernel_size = (3,3), padding = 'same', strides = 1)
        self.batch_n6 = BatchNormalization()

        self.conv_7 = Conv2D(big_conv, kernel_size = (1,1), padding = 'same', strides = 1)
        self.batch_n7 = BatchNormalization()

        self.conv_8 = Conv2D(small_conv, kernel_size = (3,3), padding = 'same', strides = 1)
        self.batch_n8 = BatchNormalization()

    def call(self, inputs):
        conv_1 = self.conv_1(inputs)
        conv_1 = self.batch_n1(conv_1)
        conv_1 = self.relu(conv_1)

        conv_2 = self.conv_2(conv_1)
        conv_2 = self.batch_n2(conv_2)
        conv_2 = self.relu(conv_2)

        concat_1 = self.concat([inputs, conv_2])

        conv_3 = self.conv_3(concat_1)
        conv_3 = self.batch_n3(conv_3)
        conv_3 = self.relu(conv_3)

        conv_4 = self.conv_4(conv_3)
        conv_4 = self.batch_n4(conv_4)
        conv_4 = self.relu(conv_4)

        concat_2 = self.concat([inputs, conv_2, conv_4])

        conv_5 = self.conv_5(concat_2)
        conv_5 = self.batch_n5(conv_5)
        conv_5 = self.relu(conv_5)

        conv_6 = self.conv_6(conv_5)
        conv_6 = self.batch_n6(conv_6)
        conv_6 = self.relu(conv_6)

        concat_3 = self.concat([inputs, conv_2, conv_4, conv_6])

        conv_7 = self.conv_7(concat_3)
        conv_7 = self.batch_n7(conv_7)
        conv_7 = self.relu(conv_7)

        conv_8 = self.conv_8(conv_7)
        conv_8 = self.batch_n8(conv_8)
        conv_8 = self.relu(conv_8)

        concat_4 = self.concat([conv_2, conv_4, conv_6, conv_8])
        
        return concat_4





#--------------------------------------------------------------------------------
# Encoding block
#--------------------------------------------------------------------------------
class DownBlock(Model):
    def __init__(self, filters, kernel_size, padding, strides):
        super(DownBlock, self).__init__()

        self.conv_1 = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = 'relu')
        self.conv_2 = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = 'relu')
        
        self.max_pool = MaxPooling2D( pool_size = (2,2), strides = (2,2) )

        # Dropout layer
        self.dropout = Dropout(0.2)


    def call(self, inputs):
        conv = self.conv_1(inputs)
        conv = self.conv_2(conv)
        dropout = self.dropout(conv)
        max_pool = self.max_pool(dropout)

        return conv, max_pool

#------------------------------------------------------------------
# Upsampling Block
#------------------------------------------------------------------
class UpBlock(Model):
    def __init__(self, filters, kernel_size, padding, strides):
        super(UpBlock, self).__init__()

        self.upsampling = UpSampling2D((2,2))
        self.concat = Concatenate()
        self.conv_1 = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = 'relu')
        self.conv_2 = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = 'relu')

        # Dropout layer
        self.dropout = Dropout(0.2)

    def call(self, inputs, skip):
        x = self.upsampling(inputs)
        x = self.concat([x, skip])
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.dropout(x)
        return x


def sigmoid_correcter(x):
    return tf.where(x > 0.5, 1.0, 0.0)

#----------------------------------------------------------------------------------------
# U-Net model
#----------------------------------------------------------------------------------------
class ResUNet(Model):
    def __init__(self, filters, kernel_size, padding, strides, with_dense):
        super(ResUNet, self).__init__()

        self.with_dense = with_dense
        self.relu = ReLU()
        if self.with_dense:
            self.dense_block_1 = DenseBlock()
            self.dense_block_2 = DenseBlock()
            self.dense_block_3 = DenseBlock()
            self.dense_block_4 = DenseBlock()

        # Create the encoding blocks
        self.down_block_1 = DownBlock(filters[0], kernel_size, padding, strides)
        self.down_block_2 = DownBlock(filters[1], kernel_size, padding, strides)
        self.down_block_3 = DownBlock(filters[2],  kernel_size, padding, strides)
        self.down_block_4 = DownBlock(filters[3], kernel_size, padding, strides)

        # Create the upsampling blocks
        self.up_block_1 = UpBlock(filters[3], kernel_size, padding, strides)
        self.up_block_2 = UpBlock(filters[2], kernel_size, padding, strides)
        self.up_block_3 = UpBlock(filters[1], kernel_size, padding, strides)
        self.up_block_4_segm = UpBlock(filters[0], kernel_size, padding, strides)
        self.up_block_4_segm_2 = UpBlock(filters[0], kernel_size, padding, strides)
        self.up_block_4_segm_3 = UpBlock(filters[0], kernel_size, padding, strides)
        self.up_block_4_segm_4 = UpBlock(filters[0], kernel_size, padding, strides)

        # Create the Residual Blocks
        self.res_block_1_1 = ResBlock(filters[1], padding = 'same', strides = 1)
        self.res_block_1_2 = ResBlock(filters[1], padding = 'same', strides = 1)
        self.res_block_1_3 = ResBlock(filters[1], padding = 'same', strides = 1)
        self.res_block_1_4 = ResBlock(filters[1], padding = 'same', strides = 1)

        self.res_block_2_1 = ResBlock(filters[2], padding = 'same', strides = 1)
        self.res_block_2_2 = ResBlock(filters[2], padding = 'same', strides = 1)
        self.res_block_2_3 = ResBlock(filters[2], padding = 'same', strides = 1)


        self.res_block_3_1 = ResBlock(filters[3], padding = 'same', strides = 1)
        self.res_block_3_2 = ResBlock(filters[3], padding = 'same', strides = 1)


        self.res_block_4 = ResBlock(filters[3], padding = 'same', strides = 1)

        # Create convolutional blocks for the bottom part
        self.conv_1 = Conv2D(filters[4], kernel_size, padding = padding, strides = strides, activation = 'relu')
        self.conv_2 = Conv2D(filters[4], kernel_size, padding = padding, strides = strides, activation = 'relu')

        self.segmenter = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name='segmenter_1')
        self.segmenter_2 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_2')
        self.segmenter_3 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_3')
        self.segmenter_4 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_4')


    def call(self, inputs):
        if self.with_dense:
            dense_1 = self.dense_block_1(inputs)
            dense_2 = self.dense_block_2(inputs)
            dense_3 = self.dense_block_3(inputs)
            dense_4 = self.dense_block_4(inputs)
        
        # Down sampling part
        c1, a1 = self.down_block_1(inputs)
        c2, a2 = self.down_block_2(a1)
        c3, a3 = self.down_block_3(a2)
        c4, a4 = self.down_block_4(a3)

        # The bottom of the U
        bottom = self.conv_1(a4)
        bottom = self.conv_2(bottom)

        # Upsampling part and residual part
        res_4 = self.res_block_4(c4)
        u1 = self.up_block_1(bottom, res_4)

        res_3 = self.res_block_3_1(c3)
        res_3 = self.res_block_3_2(res_3)
        u2 = self.up_block_2(u1, res_3)

        res_2 = self.res_block_2_1(c2)
        res_2 = self.res_block_2_2(res_2)
        res_2 = self.res_block_2_3(res_2)
        u3 = self.up_block_3(u2, res_2)

        # Top level res paths
        res_1 = self.res_block_1_1(c1)
        res_1 = self.res_block_1_2(res_1)
        res_1 = self.res_block_1_3(res_1)
        res_1 = self.res_block_1_4(res_1)

        # Add top level res paths to dense paths
        if self.with_dense:
            dense_1 = Add()([res_1, dense_1])
            dense_1 = self.relu(dense_1)
            dense_2 = Add()([res_1, dense_2])
            dense_2 = self.relu(dense_2)
            dense_3 = Add()([res_1, dense_3])
            dense_3 = self.relu(dense_3)
            dense_4 = Add()([res_1, dense_4])
            dense_4 = self.relu(dense_4)

            u4_segm = self.up_block_4_segm(u3, dense_1)
            u4_segm_2 = self.up_block_4_segm_2(u3, dense_2)
            u4_segm_3 = self.up_block_4_segm_3(u3, dense_3)
            u4_segm_4 = self.up_block_4_segm_4(u3, dense_4)
        
        else:
            u4_segm = self.up_block_4_segm(u3, res_1)
            u4_segm_2 = self.up_block_4_segm_2(u3, res_1)
            u4_segm_3 = self.up_block_4_segm_3(u3, res_1)
            u4_segm_4 = self.up_block_4_segm_4(u3, res_1)

        segm = self.segmenter(u4_segm)
        segm_2 = self.segmenter_2(u4_segm_2)
        segm_3 = self.segmenter_3(u4_segm_3)
        segm_4 = self.segmenter_4(u4_segm_4)

        return segm, segm_2, segm_3, segm_4
