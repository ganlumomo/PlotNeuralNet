
from .tikzeng import *

# define some parameters
# s_filer: output_image_size
# n_filer: output_channels
# height = depth = s_filer / SIZE_TO_HEIGHT
# width = n_filer / CHANNELS_TO_WIDTH
SIZE_TO_HEIGHT = 10
CHANNELS_TO_WIDTH = 32

#define new block
def block_IdentityResidualBlock(name, bottom, s_filer=180, n_filer=64, offset="(1,0,0)", channels=(128,128), stride=1, caption=""):
    lys = []

    is_bottleneck = len(channels) ==3
    need_proj_conv = stride != 1 or n_filer != channels[-1]

    bn1 = [to_BnRelu(name="{}_bn1".format(name), offset=offset, to="({}-east)".format(bottom), height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT, caption=caption)]

    if not is_bottleneck:
        layers = [
            to_Conv(name="{}_conv1".format(name), s_filer=s_filer, n_filer=channels[0], offset="(1,0,0)", to="({}_bn1-east)".format(name), width=channels[0]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_BnRelu(name="{}_bn2".format(name), to="({}_conv1-east)".format(name), height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
            to_Conv(name="{}_conv2".format(name), s_filer=s_filer, n_filer=channels[1], to="({}_bn2-east)".format(name), width=channels[1]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT),
        ]

    if need_proj_conv:
        proj_conv = [to_Conv(name="{}_proj_conv".format(name), s_filer=s_filer, n_filer=channels[-1], offset="(1,0,5)", to="({}_bn1-east)".format(name), width=channels[-1]/CHANNELS_TO_WIDTH, height=s_filer/SIZE_TO_HEIGHT, depth=s_filer/SIZE_TO_HEIGHT)]
        
    if need_proj_conv:
        lys += bn1
        lys += proj_conv
    else:
        pass

    if is_bottleneck:
        pass
    else:
        lys += layers

    lys += [
        to_Sum(name="{}_end".format(name), to="({}_conv2-east)".format(name))
        ]

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


