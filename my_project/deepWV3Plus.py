import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

# defined some parameters
init_size = 360
init_channels = 64

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # Input
    to_input('../examples/fcn8s/cats.jpg'),

    # Initial layers
    to_Conv("mod1", s_filer=init_size, n_filer=init_channels, width=init_channels/CHANNELS_TO_WIDTH, height=init_size/SIZE_TO_HEIGHT, depth=init_size/SIZE_TO_HEIGHT, caption="mod1"),
    to_Pool("pool2", to="(mod1-east)", height=init_size/2/SIZE_TO_HEIGHT, depth=init_size/2/SIZE_TO_HEIGHT, caption="pool2"),
    
    # Sequential MultiTask
    *block_IdentityResidualBlock("block1", bottom="pool2", s_filer=init_size/2, n_filer=init_channels),
    #to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    #to_connection( "pool1", "conv2"),
    #to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    #to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    #to_connection("pool2", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
