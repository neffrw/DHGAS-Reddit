import torch
from torch.nn import LSTM as nnLSTM
from dhgas.data.utils import make_sequences
from tqdm import tqdm

class CustomBatchLoader:
    def __init__(self, sequences, batch_size):
        self.sequences = sorted(sequences, key=len, reverse=True)
        self.batch_size = batch_size
        self.mapping = [i for i in range(len(self.sequences))]
        self.mapping.sort(key=lambda x: len(sequences[x]), reverse=True)

    def __iter__(self):
        # start = 0
        # while len(self.sequences[start]) > 5000:
        #     batch = self.sequences[start:start+1]
        #     mapping = self.mapping[start:start+1]
        #     start += 1
        #     yield batch, mapping
        for i in range(0, len(self.sequences), self.batch_size):
            batch = self.sequences[i:i + self.batch_size]
            mapping = self.mapping[i:i + self.batch_size]
            yield batch, mapping
    
    def mapping(self):
        return self.mapping

class LSTM(torch.nn.Module):
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
        self.lstm = nnLSTM(in_dim, hid_dim, num_layers, batch_first=True)
        # self.hlinear = HLinear(hid_dim, metadata, act='None')
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.nclf = nclf_linear

    def encode(self, data, *args, **kwargs):
        x_dict = self.featemb(data.x_dict)
        # x_dict = self.hlinear(x_dict)
        e_dict = data.edge_index_dict

        x_seq = make_sequences(x_dict, e_dict, self.predict_type)
        # from torch.utils.data import DataLoader
        # dataset = torch.utils.data.TensorDataset(x_seq)
        loader = CustomBatchLoader(x_seq, batch_size=512)
        # Output = empty tensor of shape x_seq.shape[0] by hid_dim
        output = torch.empty(len(x_seq), self.lstm.hidden_size, device=x_dict[self.predict_type].device)
        for batch, mapping in tqdm(loader, desc="Encoding", disable=True):
             # Get the lengths
            lengths = [len(x) for x in batch]
            # Pad sequences
            x_seq_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
            # Pack sequences
            x_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(x_seq_padded, lengths, batch_first=True)
            # batch is a list of sequences
            x, _ = self.lstm(x_seq_packed)

            # Unpack sequences
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            # Get the last output for each sequence
            lengths_tensor = torch.tensor(lengths, device=x.device)
            x = x[torch.arange(x.shape[0]), lengths_tensor - 1]
            # Get the mapping
            mapping = torch.tensor(mapping, device=x.device)
            # Map the output
            output[mapping] = x
        return output

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf(z)
        return out