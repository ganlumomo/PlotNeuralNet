
from .tikzeng import *

# define some parameters
# s_filer: output_image_size
# n_filer: output_channels
# height = depth = s_filer / SIZE_TO_HEIGHTi
# width = n_filer / CHANNELS_TO_WIDTH
SIZE_TO_HEIGHT = 10
CHANNELS_TO_WIDTH = 128

#define new block
def block_SElayerMultiTask(name, bottom, n_filer=64, offset="(0, 0, 0)", reduction=16):
  lys = []
  lys += [to_Pool(name="{}_avg_pool".format(name), offset=offset, to="({}-east)".format(bottom), width=1, height=1, depth=n_filer/CHANNELS_TO_WIDTH)]
  # SequentialMultiTask
  lys += [
    to_FcRelu(name="{}_semantic_fc1".format(name), s_filer=n_filer/reduction, offset="(0,0,-3.5)", to="({}_avg_pool-east)".format(name), depth=n_filer/reduction/CHANNELS_TO_WIDTH, caption="semantic", fill="\SemanticConvColor"),
    to_FcSigmoid(name="{}_semantic_fc2".format(name), s_filer=n_filer, to="({}_semantic_fc1-east)".format(name), depth=n_filer/CHANNELS_TO_WIDTH, fill="\SemanticConvColor"),
    to_Mul(name="{}_semantic_mul".format(name), offset="(1,0,0)", to="({}_semantic_fc2-east)".format(name)),
    to_FcRelu(name="{}_trav_fc1".format(name), s_filer=n_filer/reduction, offset="(0,0,3.5)", to="({}_avg_pool-east)".format(name), depth=n_filer/reduction/CHANNELS_TO_WIDTH, caption="traversability", fill="\TravConvColor"),
    to_FcSigmoid(name="{}_trav_fc2".format(name), s_filer=n_filer, to="({}_trav_fc1-east)".format(name), depth=n_filer/CHANNELS_TO_WIDTH, fill="\TravConvColor"),
    to_Mul(name="{}_trav_mul".format(name), offset="(1,0,0)", to="({}_trav_fc2-east)".format(name)),
  ]
  return lys


def block_IdentityResidualBlock(name, bottom, s_filer=180, n_filer=64, offset="(0.5,0,0)", channels=(128,128), stride=1, caption=""):
    lys = []

    is_bottleneck = len(channels) ==3
    need_proj_conv = stride != 1 or n_filer != channels[-1]

    bn1 = [to_BnRelu(name="{}_bn1".format(name), offset=offset, to="({}-east)".format(bottom), height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT, caption=caption)]
    lys += bn1

    if not is_bottleneck:
        layers = [
            to_Conv(name="{}_conv1".format(name), s_filer=s_filer, n_filer=channels[0], to="({}_bn1-east)".format(name), width=channels[0]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_BnRelu(name="{}_bn2".format(name), to="({}_conv1-east)".format(name), height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_Conv(name="{}_conv2".format(name), s_filer=s_filer, n_filer=channels[1], to="({}_bn2-east)".format(name), width=channels[1]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
        ]
    else:
        layers = [
            to_Conv(name="{}_conv1".format(name), s_filer=s_filer, n_filer=channels[0], to="({}_bn1-east)".format(name), width=channels[0]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_BnRelu(name="{}_bn2".format(name), to="({}_conv1-east)".format(name), height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_Conv(name="{}_conv2".format(name), s_filer=s_filer, n_filer=channels[1], to="({}_bn2-east)".format(name), width=channels[1]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_BnRelu(name="{}_bn3".format(name), to="({}_sum-east)".format(name), height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_Conv(name="{}_conv3".format(name), s_filer=s_filer, n_filer=channels[2], to="({}_bn3-east)".format(name), width=channels[2]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT)
        ]
        

    if need_proj_conv:
        proj_conv = [to_Conv(name="{}_proj_conv".format(name), s_filer=s_filer, n_filer=channels[-1], offset="(0,0,5)", to="({}_bn1-east)".format(name), width=channels[-1]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT)]
        
    if need_proj_conv:
        lys += proj_conv
    else:
        pass

    if is_bottleneck:
        pass
    else:
        lys += layers
        lys += [to_Sum(name="{}_end".format(name), to="({}_conv2-east)".format(name))]

    if is_bottleneck:
        lys += layers[0:2]
        lys += [to_Conv(name="{}_semantic".format(name), s_filer=s_filer, n_filer=channels[1], offset="(0,0,-2.5)", to="({}_bn2-east)".format(name), width=channels[1]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT, fill="\SemanticConvColor")]
        lys += [to_Conv(name="{}_traversability".format(name), s_filer=s_filer, n_filer=channels[1], offset="(0,0,2.5)", to="({}_bn2-east)".format(name), width=channels[1]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT, fill="\TravConvColor")]
        lys += layers[2]
        lys += [to_Sum(name="{}_sum".format(name), to="({}_conv2-east)".format(name))]
        lys += layers[3:]
        lys += block_SElayerMultiTask("{}_se".format(name), bottom="{}_conv3".format(name), n_filer=channels[2])
        lys += [to_Sum(name="{}_end".format(name), to="({}_se_semantic_mul-east)".format(name))]
        lys += [to_Sum(name="{}_end".format(name), to="({}_se_trav_mul-east)".format(name))]

    return lys

def block_2ConvPool( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
    to_ConvConvRelu( 
        name="ccr_{}".format( name ),
        s_filer=str(s_filer), 
        n_filer=(n_filer,n_filer), 
        offset=offset, 
        to="({}-east)".format( botton ), 
        width=(size[2],size[2]), 
        height=size[0], 
        depth=size[1],   
        ),    
    to_Pool(         
        name="{}".format( top ), 
        offset="(0,0,0)", 
        to="(ccr_{}-east)".format( name ),  
        width=1,         
        height=size[0] - int(size[0]/4), 
        depth=size[1] - int(size[0]/4), 
        opacity=opacity, ),
    to_connection( 
        "{}".format( botton ), 
        "ccr_{}".format( name )
        )
    ]


def block_Unconv( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity ),
        to_ConvRes( name='ccr_res_{}'.format(name),   offset="(0,0,0)", to="(unpool_{}-east)".format(name),    s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(ccr_res_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_ConvRes( name='ccr_res_c_{}'.format(name), offset="(0,0,0)", to="(ccr_{}-east)".format(name),       s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_res_c_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]




def block_Res( num, name, botton, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5 ):
    lys = []
    layers = [ *[ '{}_{}'.format(name,i) for i in range(num-1) ], top]
    for name in layers:        
        ly = [ to_Conv( 
            name='{}'.format(name),       
            offset=offset, 
            to="({}-east)".format( botton ),   
            s_filer=str(s_filer), 
            n_filer=str(n_filer), 
            width=size[2],
            height=size[0],
            depth=size[1]
            ),
            to_connection( 
                "{}".format( botton  ), 
                "{}".format( name ) 
                )
            ]
        botton = name
        lys+=ly
    
    lys += [
        to_skip( of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys


