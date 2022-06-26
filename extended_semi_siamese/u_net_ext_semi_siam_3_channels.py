import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, ReLU, Add


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
class ExpandedUNet(Model):
    def __init__(self, filters, kernel_size, padding, strides):
        super(ExpandedUNet, self).__init__()

        # Create the encoding blocks
        self.down_block_1 = DownBlock(filters[0], kernel_size, padding, strides)
        self.down_block_2 = DownBlock(filters[1], kernel_size, padding, strides)
        self.down_block_3 = DownBlock(filters[2],  kernel_size, padding, strides)
        self.down_block_4 = DownBlock(filters[3], kernel_size, padding, strides)

        #--------------------------------
        # Create the upsampling blocks
        #--------------------------------
        # Path 1
        self.up_block_1_1 = UpBlock(filters[3], kernel_size, padding, strides)
        self.up_block_1_2 = UpBlock(filters[2], kernel_size, padding, strides)
        self.up_block_1_3 = UpBlock(filters[1], kernel_size, padding, strides)
        self.up_block_1_4_segm = UpBlock(filters[0], kernel_size, padding, strides)
        self.segmenter = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name='segmenter_1')

        # Path 2
        self.up_block_2_1 = UpBlock(filters[3], kernel_size, padding, strides)
        self.up_block_2_2 = UpBlock(filters[2], kernel_size, padding, strides)
        self.up_block_2_3 = UpBlock(filters[1], kernel_size, padding, strides)
        self.up_block_2_4_segm = UpBlock(filters[0], kernel_size, padding, strides)
        self.segmenter_2 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_2')

        # Path 3
        self.up_block_3_1 = UpBlock(filters[3], kernel_size, padding, strides)
        self.up_block_3_2 = UpBlock(filters[2], kernel_size, padding, strides)
        self.up_block_3_3 = UpBlock(filters[1], kernel_size, padding, strides)
        self.up_block_3_4_segm = UpBlock(filters[0], kernel_size, padding, strides)
        self.segmenter_3 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_3')

        # Path 4
        self.up_block_4_1 = UpBlock(filters[3], kernel_size, padding, strides)
        self.up_block_4_2 = UpBlock(filters[2], kernel_size, padding, strides)
        self.up_block_4_3 = UpBlock(filters[1], kernel_size, padding, strides)
        self.up_block_4_4_segm = UpBlock(filters[0], kernel_size, padding, strides)
        self.segmenter_4 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_4')
        
        # Create convolutional blocks for the bottom part
        self.conv_1 = Conv2D(filters[4], kernel_size, padding = padding, strides = strides, activation = 'relu')
        self.conv_2 = Conv2D(filters[4], kernel_size, padding = padding, strides = strides, activation = 'relu')


    def call(self, inputs):
        # Down sampling part
        c1, a1 = self.down_block_1(inputs)
        c2, a2 = self.down_block_2(a1)
        c3, a3 = self.down_block_3(a2)
        c4, a4 = self.down_block_4(a3)

        # The bottom of the U
        bottom = self.conv_1(a4)
        bottom = self.conv_2(bottom)

        #-----------------------
        # Upsampling parts
        #-----------------------
        # Path 1
        u1_1 = self.up_block_1_1(bottom, c4)
        u1_2 = self.up_block_1_2(u1_1, c3)
        u1_3 = self.up_block_1_3(u1_2, c2)
        u1_4_segm = self.up_block_1_4_segm(u1_3, c1)
        segm = self.segmenter(u1_4_segm)

        # Path 2
        u2_1 = self.up_block_2_1(bottom, c4)
        u2_2 = self.up_block_2_2(u2_1, c3)
        u2_3 = self.up_block_2_3(u2_2, c2)
        u2_4_segm = self.up_block_2_4_segm(u2_3, c1)
        segm_2 = self.segmenter_2(u2_4_segm)

        # Path 3
        u3_1 = self.up_block_3_1(bottom, c4)
        u3_2 = self.up_block_3_2(u3_1, c3)
        u3_3 = self.up_block_3_3(u3_2, c2)
        u3_4_segm = self.up_block_3_4_segm(u3_3, c1)
        segm_3 = self.segmenter_3(u3_4_segm)

        # Path 4
        u4_1 = self.up_block_4_1(bottom, c4)
        u4_2 = self.up_block_4_2(u4_1, c3)
        u4_3 = self.up_block_4_3(u4_2, c2)
        u4_4_segm = self.up_block_4_4_segm(u4_3, c1)
        segm_4 = self.segmenter_4(u4_4_segm)

        return segm, segm_2, segm_3, segm_4
