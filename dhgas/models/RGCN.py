import torch
from torch_geometric.nn import RGCNConv
from dhgas.data.utils import make_hodata


class RGCN(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        metadata,
        predict_type,
        featemb=None,
        nclf_linear=None,
    ):
        super().__init__()
        num_relations = len(metadata[1])
        convs = torch.nn.ModuleList()
        convs.append(RGCNConv(in_dim, hid_dim, num_relations))

        for i in range(num_layers - 1):
            convs.append(RGCNConv(hid_dim, hid_dim, num_relations))
        self.convs = convs
        # self.hlinear = HLinear(hid_dim, metadata, act='None')
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.nclf = nclf_linear

    def encode(self, data, *args, **kwargs):
        x_dict = self.featemb(data.x_dict)
        # x_dict = self.hlinear(x_dict)
        e_dict = data.edge_index_dict

        x, e, mask, hodata = make_hodata(x_dict, e_dict, self.predict_type)
        edge_type = hodata.edge_type
        for i, conv in enumerate(self.convs):
            x = conv(x, e, edge_type)
            if i != len(self.convs) - 1:
                x = x.relu()

        if isinstance(mask, list):
            x = [x[mask[0]], x[mask[1]]]
        else:
            x = x[mask]
        return x

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf(z)
        return out
