import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout


#--------------------------------------------------------------------------------
# Inception block
#--------------------------------------------------------------------------------
class InceptionBlock(Model):
    def __init__(self, filters):
        super(InceptionBlock, self).__init__()
        self.max_pooling = MaxPooling2D(pool_size = (3,3), strides = 1, padding = 'same')

        # 1x1 convolutions
        self.conv_1 = Conv2D(filters, kernel_size = (1,1), padding = 'same', activation = 'relu', name = '1x1_1')
        self.conv_2 = Conv2D(filters, kernel_size = (1,1), padding = 'same', activation = 'relu', name = '1x1_2')
        self.conv_3 = Conv2D(filters, kernel_size = (1,1), padding = 'same', activation = 'relu', name = '1x1_3')
        self.conv_4 = Conv2D(filters, kernel_size = (1,1), padding = 'same', activation = 'relu', name = '1x1_4')

        # Other convs
        self.conv_5 = Conv2D(filters, kernel_size = (3,3), padding = 'same', activation = 'relu', name = '3x3')
        self.conv_6 = Conv2D(filters, kernel_size = (5,5), padding = 'same', activation = 'relu', name = '5x5')
        self.concat = Concatenate()
        
    def call(self, inputs):
        # First level
        conv_1 = self.conv_1(inputs)
        conv_2 = self.conv_2(inputs)
        conv_3 = self.conv_3(inputs)
        max_pool = self.max_pooling(inputs)

        # Second level
        conv_5 = self.conv_5(conv_2)
        conv_6 = self.conv_6(conv_3)
        conv_4 = self.conv_4(max_pool)

        # Third level concatenation
        concatenation = self.concat([conv_1, conv_5, conv_6, conv_4])

        return concatenation


#--------------------------------------------------------------------------------
# Encoding block
#--------------------------------------------------------------------------------
class DownBlock(Model):
    def __init__(self, filters):
        super(DownBlock, self).__init__()

        self.inception_block = InceptionBlock(filters)
        
        self.max_pool = MaxPooling2D( pool_size = (2,2), strides = (2,2) )

        # Dropout layer
        self.dropout = Dropout(0.2)


    def call(self, inputs):
        incept = self.inception_block(inputs)
        dropout = self.dropout(incept)
        max_pool = self.max_pool(dropout)

        return dropout, max_pool

#------------------------------------------------------------------
# Upsampling Block
#------------------------------------------------------------------
class UpBlock(Model):
    def __init__(self, filters):
        super(UpBlock, self).__init__()

        self.upsampling = UpSampling2D((2,2))
        self.concat = Concatenate()
        self.inception = InceptionBlock(filters)

        # Dropout layer
        self.dropout = Dropout(0.2)

    def call(self, inputs, skip):
        x = self.upsampling(inputs)
        x = self.concat([x, skip])
        x = self.inception(x)
        x = self.dropout(x)
        return x


def sigmoid_correcter(x):
    return tf.where(x > 0.5, 1.0, 0.0)

#----------------------------------------------------------------------------------------
# U-Net model
#----------------------------------------------------------------------------------------
class InceptionUNet(Model):
    def __init__(self, filters, kernel_size, padding, strides):
        super(InceptionUNet, self).__init__()

        # Create the encoding blocks
        self.down_block_1 = DownBlock(filters[0])
        self.down_block_2 = DownBlock(filters[1])
        self.down_block_3 = DownBlock(filters[2])
        self.down_block_4 = DownBlock(filters[3])

        # Create the upsampling blocks
        self.up_block_1 = UpBlock(filters[3])
        self.up_block_2 = UpBlock(filters[2])
        self.up_block_3 = UpBlock(filters[1])
        self.up_block_4_segm = UpBlock(filters[0])
        self.up_block_4_segm_2 = UpBlock(filters[0])
        self.up_block_4_segm_3 = UpBlock(filters[0])
        self.up_block_4_segm_4 = UpBlock(filters[0])

        # Create convolutional blocks for the bottom part
        self.conv_1 = Conv2D(filters[4], kernel_size, padding = padding, strides = strides, activation = 'relu')
        self.conv_2 = Conv2D(filters[4], kernel_size, padding = padding, strides = strides, activation = 'relu')

        self.segmenter = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name='segmenter_1')
        self.segmenter_2 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_2')
        self.segmenter_3 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_3')
        self.segmenter_4 = Conv2D( 1, (1,1), padding = 'same', strides =1, activation = 'sigmoid', name = 'segmenter_4')

    def call(self, inputs):
        # Down sampling part
        c1, a1 = self.down_block_1(inputs)
        c2, a2 = self.down_block_2(a1)
        c3, a3 = self.down_block_3(a2)
        c4, a4 = self.down_block_4(a3)

        # The bottom of the U
        bottom = self.conv_1(a4)
        bottom = self.conv_2(bottom)

        # Upsampling part
        u1 = self.up_block_1(bottom, c4)
        u2 = self.up_block_2(u1, c3)
        u3 = self.up_block_3(u2, c2)
        u4_segm = self.up_block_4_segm(u3, c1)
        u4_segm_2 = self.up_block_4_segm_2(u3, c1)
        u4_segm_3 = self.up_block_4_segm_3(u3, c1)
        u4_segm_4 = self.up_block_4_segm_4(u3, c1)

        segm = self.segmenter(u4_segm)
        segm_2 = self.segmenter_2(u4_segm_2)
        segm_3 = self.segmenter_3(u4_segm_3)
        segm_4 = self.segmenter_4(u4_segm_4)

        return segm, segm_2, segm_3, segm_4
