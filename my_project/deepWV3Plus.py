import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

# defined some parameters
init_size = 360
init_channels = 64
structure = [3, 3, 6, 3, 1, 1]
channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048,4096)]
# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # Input
    to_input('../examples/fcn8s/cats.jpg'),

    # Initial layers
    to_Conv("mod1", s_filer=init_size, n_filer=init_channels, width=init_channels/CHANNELS_TO_WIDTH, height=init_size/SIZE_TO_HEIGHT, depth=init_size/SIZE_TO_HEIGHT, caption="mod1"),
    
    #*block_IdentityResidualBlock("block1", bottom="pool2", s_filer=init_size/2, n_filer=init_channels),
    #*block_IdentityResidualBlock("block2", bottom="block1_sum", s_filer=init_size/2, n_filer=)
    #to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    #to_connection( "pool1", "conv2"),
    #to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    #to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    #to_connection("pool2", "soft1"),
    #to_end()
    ]
for mod_id, num in enumerate(structure):
  if mod_id < 2:
    arch += [to_Pool("pool{}".format(mod_id+2), to="(mod1-east)", height=init_size/2/SIZE_TO_HEIGHT, depth=init_size/2/SIZE_TO_HEIGHT, caption="pool{}".format(mod_id+2)),]
  for block_id in range(num):
    if block_id == 0:
      arch += [*block_IdentityResidualBlock("m{}_b{}".format(mod_id+2, block_id+1), bottom="pool2", s_filer=init_size/2, n_filer=init_channels),]
    else:
      arch += [*block_IdentityResidualBlock("m{}_b{}".format(mod_id+2, block_id+1), bottom="m{}_b{}_end".format(mod_id+2, block_id), s_filer=init_size/2, n_filer=init_channels),]


arch += [to_end(),]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
