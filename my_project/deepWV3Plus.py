import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

# defined some parameters
init_size = 360
init_channels = 64
class_num = 30
structure = [3, 3, 6, 3, 1, 1]
channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048,4096)]
#dilations = [1, 1, 1, 2, 4, 4]
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

current_size = init_size
for mod_id, num in enumerate(structure):
  if mod_id == 0:
    current_size /= 2 # because of pool1
    arch += [to_Pool("pool{}".format(mod_id+2), to="(mod1-east)", height=current_size/SIZE_TO_HEIGHT, depth=current_size/SIZE_TO_HEIGHT, caption="pool{}".format(mod_id+2)),]
  if mod_id == 1:
    current_size /= 2 # becasue of pool2
    arch += [to_Pool("pool{}".format(mod_id+2), to="(m2_end_end-east)", height=current_size/SIZE_TO_HEIGHT, depth=current_size/SIZE_TO_HEIGHT, caption="pool{}".format(mod_id+2)),]

  in_channels = init_channels
  for block_id in range(num):
    stride = 2 if block_id == 0 and mod_id == 2 else 1
    current_size /= 2 if block_id == 0 and mod_id == 2 else 1
    if block_id == num-1:
      if mod_id > 3:
        arch += [*block_IdentityResidualBlock("m{}_end".format(mod_id+2), offset="(0,0,{})".format(-3.5*(mod_id-4)), bottom="m{}_end_end".format(mod_id+1), s_filer=current_size, n_filer=in_channels, channels=channels[mod_id], stride=stride),]
      else:
        arch += [*block_IdentityResidualBlock("m{}_end".format(mod_id+2), bottom="m{}_b{}_end".format(mod_id+2, block_id), s_filer=current_size, n_filer=in_channels, channels=channels[mod_id], stride=stride),]
    elif block_id == 0:
      if mod_id == 0:
        arch += [*block_IdentityResidualBlock("m{}_b{}".format(mod_id+2, block_id+1), bottom="pool2", s_filer=current_size, n_filer=in_channels, channels=channels[mod_id], stride=stride, caption="mod{}".format(mod_id+2)),]
      else:
        arch += [*block_IdentityResidualBlock("m{}_b{}".format(mod_id+2, block_id+1), bottom="m{}_end_end".format(mod_id+1), s_filer=current_size, n_filer=in_channels, channels=channels[mod_id], stride=stride, caption="mod{}".format(mod_id+2)),]
    else:
      arch += [*block_IdentityResidualBlock("m{}_b{}".format(mod_id+2, block_id+1), bottom="m{}_b{}_end".format(mod_id+2, block_id), s_filer=current_size, n_filer=in_channels, channels=channels[mod_id], stride=stride),]
    in_channels = channels[mod_id][-1]

# Task-Specific Decoder
arch += [
  to_Conv("aspp_semantic", s_filer=current_size, n_filer=1280, to="(m7_end_se_semantic_mul-east)", width=1280/CHANNELS_TO_WIDTH, height=current_size/SIZE_TO_HEIGHT, depth=current_size/SIZE_TO_HEIGHT, fill="\SemanticConvColor"),
  to_Conv("bot_aspp_semantic", s_filer=current_size, n_filer=256, to="(aspp_semantic-east)", width=256/CHANNELS_TO_WIDTH, height=current_size/SIZE_TO_HEIGHT, depth=current_size/SIZE_TO_HEIGHT, fill="\SemanticConvColor"),
  to_Conv("bot_fine_semantic", s_filer=180, n_filer=48, to="(bot_aspp_semantic-east)", width=48/CHANNELS_TO_WIDTH, height=180/SIZE_TO_HEIGHT, depth=180/SIZE_TO_HEIGHT, fill="\SemanticConvColor"),
  to_Conv("upsample1_semantic", s_filer=180, n_filer=256, to="(bot_fine_semantic-east)", width=256/CHANNELS_TO_WIDTH, height=180/SIZE_TO_HEIGHT, depth=180/SIZE_TO_HEIGHT, fill="\SemanticConvColor"),
  to_Conv("final_semantic", s_filer=180, n_filer=class_num, offset="(0.5,0,0)", to="(upsample1_semantic-east)", width=class_num/CHANNELS_TO_WIDTH, height=180/SIZE_TO_HEIGHT, depth=180/SIZE_TO_HEIGHT, fill="\SemanticConvColor"),
  to_Conv("upsample2_semantic", s_filer=init_size, n_filer=class_num, offset="(0.5,0,0)", to="(final_semantic-east)", width=class_num/CHANNELS_TO_WIDTH, height=init_size/SIZE_TO_HEIGHT, depth=init_size/SIZE_TO_HEIGHT, fill="\SemanticConvColor"),
  to_connection_orthogonal(of="m2_end_end", to="bot_fine_semantic", offset="(0,0,-3.5)"),

  to_Conv("aspp_trav", s_filer=current_size, n_filer=1280, to="(m7_end_se_trav_mul-east)", width=1280/CHANNELS_TO_WIDTH, height=current_size/SIZE_TO_HEIGHT, depth=current_size/SIZE_TO_HEIGHT, fill="\TravConvColor"),
  to_Conv("bot_aspp_trav", s_filer=current_size, n_filer=256, to="(aspp_trav-east)", width=256/CHANNELS_TO_WIDTH, height=current_size/SIZE_TO_HEIGHT, depth=current_size/SIZE_TO_HEIGHT, fill="\TravConvColor"),
  to_Conv("bot_fine_trav", s_filer=180, n_filer=48, to="(bot_aspp_trav-east)", width=48/CHANNELS_TO_WIDTH, height=180/SIZE_TO_HEIGHT, depth=180/SIZE_TO_HEIGHT, fill="\TravConvColor"),
  to_Conv("upsample1_trav", s_filer=180, n_filer=256, to="(bot_fine_trav-east)", width=256/CHANNELS_TO_WIDTH, height=180/SIZE_TO_HEIGHT, depth=180/SIZE_TO_HEIGHT, fill="\TravConvColor"),
  to_Conv("final_trav", s_filer=180, n_filer=2, offset="(0.5,0,0)", to="(upsample1_trav-east)", width=2/CHANNELS_TO_WIDTH, height=180/SIZE_TO_HEIGHT, depth=180/SIZE_TO_HEIGHT, fill="\TravConvColor"),
  to_Conv("upsample2_trav", s_filer=init_size, n_filer=2, offset="(0.5,0,0)", to="(final_trav-east)", width=class_num/CHANNELS_TO_WIDTH, height=init_size/SIZE_TO_HEIGHT, depth=init_size/SIZE_TO_HEIGHT, fill="\TravConvColor"),
  to_connection_orthogonal(of="m2_end_end", to="bot_fine_trav", offset="(0,0,3.5)"),

  to_end()
  ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
