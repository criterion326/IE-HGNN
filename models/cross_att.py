import sys
import datetime

LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR']
LEVELS_MAP = None

def init_map():
    global LEVELS_MAP, LEVELS
    LEVELS_MAP = {}
    for idx, level in enumerate(LEVELS):
        LEVELS_MAP[level] = idx
def get_prio(level):
    global LEVELS_MAP
    if LEVELS_MAP is None:
        init_map()
    return LEVELS_MAP[level.upper()]
def print_log(s, level='INFO', end='\n', no_prefix=False):
    pth_prio = get_prio(os.getenv('LOG', 'INFO'))
    prio = get_prio(level)
    if prio >= pth_prio:
        if not no_prefix:
            now = datetime.datetime.now()
            prefix = now.strftime("%Y-%m-%d %H:%M:%S") + f'::{level.upper()}::'
            print(prefix, end='')
        print(s, end=end)
        sys.stdout.flush()

def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result
  
def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    # print('coord_diff',coord_diff.shape)
    radial = th.bmm(coord_diff,
                   coord_diff.transpose(-1,
                                        -2))  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, coord_diff
  
class CrossAttentionLayer(nn.Module):
    """
    Cross Attention Layer
    """

    def __init__(self,
                 input_nf,
                 output_nf,
                 hidden_nf,
                 n_channel,
                 edges_in_d=0,
                 act_fn=nn.SiLU(),
                 dropout=0.1):
        super().__init__()
        self.hidden_nf = hidden_nf

        self.dropout = nn.Dropout(dropout)

        self.linear_q = nn.Linear(input_nf, hidden_nf)
        self.linear_kv = nn.Linear(input_nf + n_channel**2 + edges_in_d,
                                   hidden_nf * 2)  # parallel calculate kv

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def att_model(self, h1,h2, edge_index, radial, edge_attr):
        '''
        :param h1: [bs * n_node, input_size]
        :param h2: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param radial: [n_edge, n_channel, n_channel]
        :param edge_attr: [n_edge, edge_dim]
        '''
        row, col = edge_index
        # print('g_shape',h.shape)
        # source, target = h[row], h[col]  # [n_edge, input_size]
        source,target=h1,h2
        # qkv
        q = self.linear_q(source)  # [n_edge, hidden_size]
        # print('radial',radial.shape)
        n_channel = radial.shape[1]
        # print('n_channel',n_channel)
        radial = radial.reshape(radial.shape[0], n_channel *
                                n_channel)  # [n_edge, n_channel ^ 2]
        # print('102',radial.shape,target.shape,edge_attr.shape)
        if edge_attr is not None:
            target_feat = torch.cat([radial, target, edge_attr], dim=1)
        else:
            target_feat = torch.cat([radial, target], dim=1)
        kv = self.linear_kv(target_feat)  # [n_edge, hidden_size * 2]
        k, v = kv[..., 0::2], kv[..., 1::2]  # [n_edge, hidden_size]

        # attention weight
        alpha = torch.sum(q * k, dim=1)  # [n_edge]

        print_log(f'alpha1, {torch.isnan(alpha).sum()}', level='DEBUG')

        # alpha = scatter_softmax(alpha, row, h.shape[0]) # [n_edge]
        alpha = scatter_softmax(alpha, row)  # [n_edge]

        print_log(f'alpha2, {torch.isnan(alpha).sum()}', level='DEBUG')

        return alpha, v

    def node_model(self, h, edge_index, att_weight, v):
        '''
        :param h: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param att_weight: [n_edge, 1], unsqueezed before passed in
        :param v: [n_edge, hidden_size]
        '''
        row, _ = edge_index
        agg = unsorted_segment_sum(att_weight * v, row,
                                   h.shape[0])  # [bs * n_node, hidden_size]
        agg = self.dropout(agg)
        return h + agg  # residual 残差，注意力聚合得到的消息更新h

    def coord_model(self, coord, edge_index, coord_diff, att_weight, v):
        '''
        :param coord: [bs * n_node, n_channel, d]
        :param edge_index: list of [n_edge], [n_edge]
        :param coord_diff: [n_edge, n_channel, d]
        :param att_weight: [n_edge, 1], unsqueezed before passed in
        :param v: [n_edge, hidden_size]
        '''
        row, _ = edge_index
        coord_v = att_weight * self.coord_mlp(v)  # [n_edge, n_channel]
        trans = coord_diff * coord_v.unsqueeze(-1)  # [n_edge,n_channel,d]
        agg = unsorted_segment_sum(trans, row, coord.size(0))
        coord = coord + agg
        return coord  #残差更新坐标

    def forward(self, h, edge_index, coord, edge_attr=None):
        radial, coord_diff = coord2radial(edge_index, coord)  #153,1  153,3

        att_weight, v = self.att_model(h, edge_index, radial, edge_attr)

        print_log(f'att_weight, {torch.isnan(att_weight).sum()}',
                  level='DEBUG')
        print_log(f'v, {torch.isnan(v).sum()}', level='DEBUG')

        flat_att_weight = att_weight
        att_weight = att_weight.unsqueeze(-1)  # [n_edge, 1]
        h = self.node_model(h, edge_index, att_weight, v)
        coord = self.coord_model(coord, edge_index, coord_diff, att_weight, v)
        # h = global_mean_pool(h,batch=) #先不做全局池化
        # h=th.mean(h,dim=0,keepdim=True)
        return h, coord, flat_att_weight
