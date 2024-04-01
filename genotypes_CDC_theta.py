from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    # 'none',
    'max_pool_3x3',
    'SepCDC_3x3_0.0',
    'SepCDC_3x3_0.5',
    'SepCDC_3x3_0.7',
    'SepCDC_3x3_0.8',
    'DilCDC_3x3_0.0',
    'DilCDC_3x3_0.5',
    'DilCDC_3x3_0.7',
    'DilCDC_3x3_0.8',
    'skip_connect',
    'none'
]



NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


PCDARTS = PC_DARTS_cifar

# SEARCH_NET = Genotype(
#         normal=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0),
#                 ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6),
#         reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 3),
#                 ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))





# SEARCH_NET = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0),
#                               ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)],
#                       normal_concat=range(2, 6),
#                       reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2),
#                               ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 4)],
#                       reduce_concat=range(2, 6))


# SEARCH_NET = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2),
#                               ('max_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 1)],
#                       normal_concat=range(2, 6),
#                       reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1),
#                               ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 3)],
#                       reduce_concat=range(2, 6))


SEARCH_NET_20220505 = Genotype(normal=[('inverse_residual_k5_e6_g1', 1), ('inverse_residual_k3_e1_g2_d2', 0),
                                       ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 2),
                                       ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
                               reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
                                       ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
                                       ('max_pool_3x3', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

SEARCH_NET_0506 = Genotype(normal=[('inverse_residual_k5_e6_g1', 1), ('inverse_residual_k3_e1_g2_d2', 0),
                                       ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 2),
                                       ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
                               reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
                                       ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
                                       ('max_pool_3x3', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))


SEARCH_NET_0510 = Genotype(normal=[('inverse_residual_k3_e3_g1_d2', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2),
                                   ('inverse_residual_k3_e1_g1', 0), ('sep_conv_3x3', 3), ('max_pool_3x3', 1),
                                   ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6),
                           reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1),
                                   ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
                                   ('max_pool_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))


SEARCH_NET_0520 = Genotype(normal=[('dil_conv_3x3', 0), ('Bayar', 1), ('sep_conv_3x3', 2),
                                   ('dil_conv_3x3', 0), ('Bayar', 2), ('dil_conv_3x3', 0),
                                   ('sep_conv_3x3', 4), ('Bayar', 0)], normal_concat=range(2, 6),
                           reduce=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1),
                                   ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0),
                                   ('Bayar', 2), ('Bayar', 4)], reduce_concat=range(2, 6))
SEARCH_NET_0522 = Genotype(normal=[('sep_conv_3x3', 0), ('Bayar', 1), ('dil_conv_5x5', 0),
                                   ('Bayar', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1),
                                   ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
                           reduce=[('Bayar', 0), ('avg_pool_3x3', 1), ('skip_connect', 2),
                                   ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0),
                                   ('avg_pool_3x3', 4), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

SEARCH_NET_0525 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0),
                                   ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0),
                                   ('dil_conv_3x3', 4), ('avg_pool_3x3', 0)], normal_concat=range(2, 6),
                           reduce=[('Bayar', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('Bayar', 2),
                                   ('Bayar', 2), ('max_pool_3x3', 3), ('Bayar', 2), ('max_pool_3x3', 3)],
                           reduce_concat=range(2, 6))

SEARCH_NET_0531 = Genotype(normal=[('inverse_residual_k3_e1_g1_d2', 1), ('skip_connect', 0), ('inverse_residual_k3_e3_g1', 2),
                                   ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('inverse_residual_k3_e6_g1_d2', 3),
                                   ('skip_connect', 0), ('Bayar', 1)], normal_concat=range(2, 6),
                           reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0),
                                   ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('SRM', 1), ('Bayar', 0),
                                   ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
SEARCH_NET_0929 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0),
                                   ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3),
                                   ('dil_conv_5x5', 2), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
                           reduce=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2),
                                   ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3),
                                   ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
SEARCH_NET_0929_2 = Genotype(normal=[('DilCDC_3x3_0.7', 1), ('SepCDC_3x3_0.8', 0), ('max_pool_3x3', 2),
                                     ('SepCDC_3x3_0.0', 1), ('SepCDC_3x3_0.5', 2), ('SepCDC_3x3_0.0', 3),
                                     ('SepCDC_3x3_0.0', 4), ('DilCDC_3x3_0.8', 2)], normal_concat=range(2, 6),
                             reduce=[('SepCDC_3x3_0.8', 0), ('SepCDC_3x3_0.8', 1), ('SepCDC_3x3_0.8', 1),
                                     ('SepCDC_3x3_0.7', 2), ('DilCDC_3x3_0.7', 0), ('SepCDC_3x3_0.5', 1),
                                     ('DilCDC_3x3_0.8', 1), ('SepCDC_3x3_0.8', 0)], reduce_concat=range(2, 6))

SEARCH_NET_0930 = Genotype(normal=[('SepCDC_3x3_0.8', 1), ('SepCDC_3x3_0.8', 0),
                                   ('SepCDC_3x3_0.5', 1), ('SepCDC_3x3_0.8', 2),
                                   ('max_pool_3x3', 1), ('max_pool_3x3', 3),
                                   ('max_pool_3x3', 4), ('max_pool_3x3', 3)],
                           normal_concat=range(2, 6),
                           reduce=[('SepCDC_3x3_0.8', 1), ('SepCDC_3x3_0.8', 0),
                                   ('SepCDC_3x3_0.8', 2), ('SepCDC_3x3_0.7', 1),
                                   ('DilCDC_3x3_0.5', 2), ('DilCDC_3x3_0.5', 3),
                                   ('SepCDC_3x3_0.5', 1), ('DilCDC_3x3_0.5', 3)],
                           reduce_concat=range(2, 6))

SEARCH_NET_1007 = Genotype(normal=[('SepCDC_3x3_0.5', 1), ('max_pool_3x3', 0),
                                   ('SepCDC_3x3_0.5', 2), ('SepCDC_3x3_0.8', 1),
                                   ('SepCDC_3x3_0.7', 2), ('SepCDC_3x3_0.5', 3),
                                   ('SepCDC_3x3_0.0', 4), ('SepCDC_3x3_0.0', 3)],
                           normal_concat=range(2, 6),
                           reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0),
                                   ('max_pool_3x3', 1), ('SepCDC_3x3_0.7', 2),
                                   ('SepCDC_3x3_0.8', 2), ('max_pool_3x3', 1),
                                   ('SepCDC_3x3_0.8', 2), ('max_pool_3x3', 1)],
                           reduce_concat=range(2, 6))
SEARCH_NET_1010 = Genotype(normal=[('SepCDC_3x3_0.7', 1), ('SepCDC_3x3_0.8', 0),
                                   ('max_pool_3x3', 1), ('SepCDC_3x3_0.0', 2),
                                   ('SepCDC_3x3_0.5', 2), ('SepCDC_3x3_0.0', 0),
                                   ('SepCDC_3x3_0.5', 4), ('SepCDC_3x3_0.7', 3)],
                           normal_concat=range(2, 6),
                           reduce=[('SepCDC_3x3_0.8', 1), ('SepCDC_3x3_0.7', 0),
                                   ('SepCDC_3x3_0.7', 2), ('SepCDC_3x3_0.7', 0),
                                   ('SepCDC_3x3_0.5', 2), ('SepCDC_3x3_0.8', 1),
                                   ('SepCDC_3x3_0.8', 1), ('SepCDC_3x3_0.5', 2)],
                           reduce_concat=range(2, 6))

SEARCH_NET_0930 = Genotype(normal=[('SepCDC_3x3_0.8', 1), ('SepCDC_3x3_0.8', 0),
                                   ('SepCDC_3x3_0.5', 1), ('SepCDC_3x3_0.8', 2),
                                   ('max_pool_3x3', 1), ('max_pool_3x3', 3),
                                   ('max_pool_3x3', 4), ('max_pool_3x3', 3)],
                           normal_concat=range(2, 6),
                           reduce=[('SepCDC_3x3_0.8', 1), ('SepCDC_3x3_0.8', 0),
                                   ('SepCDC_3x3_0.8', 2), ('SepCDC_3x3_0.7', 1),
                                   ('DilCDC_3x3_0.5', 2), ('DilCDC_3x3_0.5', 3),
                                   ('SepCDC_3x3_0.5', 1), ('DilCDC_3x3_0.5', 3)],
                           reduce_concat=range(2, 6))

SEARCH_NET_1102 = Genotype(normal=[('SepCDC_3x3_0.0', 1), ('SepCDC_3x3_0.5', 0),
                                   ('SepCDC_3x3_0.0', 2), ('SepCDC_3x3_0.8', 1),
                                   ('SepCDC_3x3_0.7', 2), ('SepCDC_3x3_0.7', 1),
                                   ('SepCDC_3x3_0.7', 1), ('SepCDC_3x3_0.0', 2)],
                           normal_concat=range(2, 6),
                           reduce=[('SepCDC_3x3_0.8', 0), ('SepCDC_3x3_0.7', 1),
                                   ('SepCDC_3x3_0.5', 1), ('SepCDC_3x3_0.7', 2),
                                   ('SepCDC_3x3_0.7', 2), ('SepCDC_3x3_0.7', 3),
                                   ('max_pool_3x3', 4), ('SepCDC_3x3_0.7', 3)],
                           reduce_concat=range(2, 6))
