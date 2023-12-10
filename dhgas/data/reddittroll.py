from dhgas.data.utils import time_select_edge_time as time_select
import torch_geometric.transforms as T
from dhgas.utils import move_to, setup_seed
import random
import numpy as np
from dhgas.data.utils import *
from torch_geometric.data import HeteroData, Data
import torch as th
from torch import Tensor, LongTensor, FloatTensor
from torch_geometric.transforms import ToUndirected
from dhgas.data.utils import time_merge_edge_time
from dhgas import config
import os.path as osp
import pandas as pd

dataroot = osp.join(config.dataroot, "Reddit-troll")
author_post_file = osp.join(dataroot, "RedditTroll_author-post.csv")
post_post_file = osp.join(dataroot, "RedditTroll_post-post.csv")
# post_subreddit_file = osp.join(dataroot, "RedditTroll_post-subreddit.csv")
post_suspicious_file = osp.join(dataroot, "RedditTroll_suspicious-names.csv")
post_body_embedding_files = [
    "0-hop-submissions_embeddings_final.pt",
    "1-hop_submissions_embeddings_final.pt",
    "0-hop-comments_embeddings_final.pt",
    "1-hop_comments_embeddings_final.pt",
]
post_body_embedding_files = [osp.join(dataroot, f) for f in post_body_embedding_files]
extracted_data_files = [
    "0-hop-submissions.csv",
    "1-hop_submissions.csv",
    "0-hop-comments.csv",
    "1-hop_comments.csv",
]
extracted_data_files = [osp.join(dataroot, f) for f in extracted_data_files]


class RedditTrollDataset:
    def __init__(self, undirected=True):
        # 1. read data
        df_author_post = pd.read_csv(author_post_file)
        nauthors = len(df_author_post["author"].unique())
        # df_author_post["edge_type"] = 0
        # Remove everything to the left of "_" in "id", if there is a "_"
        df_author_post["id"] = df_author_post["id"].apply(
            lambda x: x.split("_")[1] if "_" in x else x
        )
        df_author_post["src"] = df_author_post["author"]
        df_author_post["dst"] = df_author_post["id"]
        df_post_post = pd.read_csv(post_post_file)
        # df_post_post["edge_type"] = 1
        # remove empty parent ids
        df_post_post["parent_id"].fillna("", inplace=True)
        df_post_post = df_post_post[df_post_post["parent_id"] != ""]
        nposts = len(df_author_post["id"])
        # Remove everything to the left of "_" in "id", if there is a "_"
        df_post_post["id"] = df_post_post["id"].apply(
            lambda x: x.split("_")[1] if "_" in x else x
        )
        df_post_post["parent_id"] = df_post_post["parent_id"].apply(
            lambda x: x.split("_")[1] if "_" in x else x
        )
        df_post_post["src"] = df_post_post["id"]
        df_post_post["dst"] = df_post_post["parent_id"]
        # df_post_post.columns("id parent_id".split())
        # df_post_subreddit = pd.read_csv(post_subreddit_file)
        # df_post_subreddit["edge_type"] = 2
        # nsubreddits = len(df_post_subreddit["subreddit"].unique())
        # df = pd.concat([df_author_post, df_post_post, df_post_subreddit])
        # df = pd.concat([df_author_post, df_post_post])
        # 2. reorder index
        # Create mapping of authors to new indices
        author_to_idx = {
            author: idx for idx, author in enumerate(df_author_post["author"].unique())
        }
        # Print nunique authors
        # print(f"nunique authors: {len(author_to_idx)}")
        # Create mapping of posts to new indices
        post_to_idx = {
            post: idx for idx, post in enumerate(df_author_post["id"].unique())
        }

        # Create mapping of times to new indices, in order
        time_list = df_author_post["datetime"].unique().tolist()
        time_list.sort()
        time_to_idx = {time: idx for idx, time in enumerate(time_list)}
        # Create mapping of subreddits to new indices
        # subreddit_to_idx = {
        #     subreddit: idx
        #     for idx, subreddit in enumerate(df_post_subreddit["subreddit"].unique())
        # }
        df_author_post["src"] = df_author_post["src"].map(author_to_idx)
        # Print nunique src
        # print(f"nunique src: {len(df_author_post['src'].unique())}")
        df_author_post["dst"] = df_author_post["dst"].map(post_to_idx)
        # df_author_post["datetime"] = df_author_post["datetime"].map(time_to_idx)
        # map df_post_post["src"] to post_to_idx
        df_post_post["src"] = df_post_post["src"].map(post_to_idx)
        df_post_post["dst"] = df_post_post["dst"].map(post_to_idx)
        # df_post_post["datetime"] = df_post_post["datetime"].map(time_to_idx)
        # Remove rows with NaN
        df_post_post.dropna(inplace=True)
        # data[:, 2] = [subreddit_to_idx[subreddit] for subreddit in data[:, 2]]
        # 3. statistics
        self.times = df_author_post["datetime"].unique().tolist()
        self.times.sort()
        # three types
        etypes = [
            tuple("author compose post".split()),
            tuple("post reply post".split()),
            # tuple("post within subreddit".split()),
        ]
        # 4 put in HeteroData
        dataset = HeteroData()
        for etype, df in zip(etypes, [df_author_post, df_post_post]):
            dataset[etype].edge_index = LongTensor(df[["src", "dst"]].to_numpy().T)
            # Print nunique src, if etype = "author compose post"
            # if etype == tuple("author compose post".split()):
            #     print(f"nunique src: {len(df['src'].unique())}")
            dataset[etype].edge_time = LongTensor(df["datetime"].to_numpy())

        # dataset["author"].x = th.arange(0, nauthors, dtype=th.float).unsqueeze(-1)
        dataset["author"].x = th.zeros(nauthors, dtype=th.float).unsqueeze(-1)
        post_embeddings = [torch.load(f) for f in post_body_embedding_files]
        post_embeddings = torch.cat(post_embeddings, dim=0)
        assert post_embeddings.shape[0] == nposts
        post_metrics = [pd.read_csv(f) for f in extracted_data_files]
        post_metrics = pd.concat(post_metrics, axis=0)
        post_metrics = post_metrics[["edited", "score"]]
        post_metrics["edited"] = post_metrics["edited"].astype(bool)
        post_metrics["edited"] = post_metrics["edited"].astype(int)
        post_metrics = FloatTensor(post_metrics.to_numpy())
        assert post_metrics.shape[0] == nposts
        # Add columns to post_embeddings from post_metrics
        post_embeddings = torch.cat(
            [post_embeddings.to(post_metrics.device), post_metrics], dim=1
        )
        assert post_embeddings.shape[0] == nposts
        dataset["post"].x = post_embeddings
        # dataset["post"].x = th.arange(0, nposts).unsqueeze(-1)
        # dataset["subreddit"].x = th.arange(0, nsubreddits).unsqueeze(-1)

        df_suspicious = pd.read_csv(post_suspicious_file)
        dataset["author"].y = FloatTensor(
            [
                1 if author in df_suspicious["Username"].unique() else 0
                for author in df_author_post["author"].unique()
            ]
        ).unsqueeze(-1)

        if undirected:
            dataset = ToUndirected()(dataset)

        self.dataset = dataset

    def times(self):
        return self.times


def hetero_remove_edges_unseen_nodes(data, etype, train_nodes0, train_nodes1):
    """inplace operation, remove edges with nodes not in train_nodes"""
    idxs = []
    ei = data[etype].edge_index.T  # [E,2]
    # print(f'before removing : {ei.T.shape}')
    for i in range(ei.shape[0]):
        e = ei[i].numpy()
        if (e[0] in train_nodes0) and (e[1] in train_nodes1):
            idxs.append(i)
    idxs = torch.LongTensor(idxs)
    data[etype].edge_index = torch.index_select(data[etype].edge_index, 1, idxs)
    # print(f'after removing : {data[etype].edge_index.shape}')


def train_val_test_split(maxn, val_ratio=0.1, test_ratio=0.1, positive_label_mask=None):
    assert positive_label_mask is not None

    # Ensure a similar ratio of positive and negative examples in each split.
    # This is done by first randomly selecting positive examples, and then
    # randomly selecting negative examples.
    # Convert positive_label_mask to a boolean mask if it is not already.
    if positive_label_mask.dtype != torch.bool:
        positive_label_mask = positive_label_mask.bool()
    positive_idxs = torch.nonzero(positive_label_mask).squeeze(-1)
    negative_idxs = torch.nonzero(~positive_label_mask).squeeze(-1)
    positive_num = positive_idxs.shape[0]
    negative_num = negative_idxs.shape[0]
    assert positive_num + negative_num == maxn
    # Shuffle positive and negative examples.
    positive_idxs = positive_idxs[torch.randperm(positive_num)]
    negative_idxs = negative_idxs[torch.randperm(negative_num)]

    val_pos_num = int(np.ceil(val_ratio * positive_num))
    val_neg_num = int(np.ceil(val_ratio * negative_num))
    test_pos_num = int(np.ceil(test_ratio * positive_num))
    test_neg_num = int(np.ceil(test_ratio * negative_num))
    train_pos_num = positive_num - val_pos_num - test_pos_num
    train_neg_num = negative_num - val_neg_num - test_neg_num
    assert train_pos_num >= 0 and test_pos_num >= 0 and val_pos_num >= 0
    assert train_neg_num >= 0 and test_neg_num >= 0 and val_neg_num >= 0
    assert positive_num

    test_idxs = torch.cat((positive_idxs[:test_pos_num], negative_idxs[:test_neg_num]))
    val_idxs = torch.cat(
        (
            positive_idxs[test_pos_num : test_pos_num + val_pos_num],
            negative_idxs[test_neg_num : test_neg_num + val_neg_num],
        )
    )
    train_idxs = torch.cat(
        (
            positive_idxs[test_pos_num + val_pos_num :],
            negative_idxs[test_neg_num + val_neg_num :],
        )
    )
    # print(f"split sizes: train {train_num} ; val {val_num} ; test {test_num}")
    print(
        f"split sizes: train {train_idxs.shape[0]} ; val {val_idxs.shape[0]} ; test {test_idxs.shape[0]}"
    )

    train_mask = torch.zeros(maxn).bool()
    train_mask[train_idxs] = True
    val_mask = torch.zeros(maxn).bool()
    val_mask[val_idxs] = True
    test_mask = torch.zeros(maxn).bool()
    test_mask[test_idxs] = True

    return train_mask, val_mask, test_mask


class RedditTrollUniDataset:
    def __init__(
        self,
        time_window=1,
        shuffle=True,
        test_full=False,
        is_dynamic=True,
        seed=22,
        val_ratio=0.1,
        test_ratio=0.1,
    ):
        self.time_window = time_window
        self.shuffle = shuffle
        self.is_dynamic = is_dynamic

        setup_seed(seed)  # seed preprocess
        dataset = RedditTrollDataset(undirected=True)

        setup_seed(seed)  # seed spliting
        times = dataset.times  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = dataset.dataset  # Hetero

        datas = [time_select(dataset, i) for i in times]  # heteros

        def get_eval_data(dataset, mask):
            eval_data = Data()
            eval_data.y = dataset["author"].y
            eval_data.mask = mask
            return eval_data

        train_mask, val_mask, test_mask = train_val_test_split(
            len(dataset["author"].x),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            positive_label_mask=dataset["author"].y,
        )

        train_eval = get_eval_data(dataset, train_mask)
        val_eval = get_eval_data(dataset, val_mask)
        test_eval = get_eval_data(dataset, test_mask)

        self.eval_datas = [train_eval, val_eval, test_eval]
        self.datas = datas
        self.dataset = dataset
        self.metadata = dataset.metadata()

        if is_dynamic:
            time_merge = lambda x: x
        else:
            time_merge = time_merge_edge_time

        # maxn = largest edge_time - smallest edge_time
        # print(datas)
        # maxn = len(times)
        # patchlen = maxn // time_window
        # self.time_dataset = time_merge(
        #     [
        #         time_merge_edge_time([datas[i] for i in range(k, k + patchlen)])
        #         for k in range(0, maxn, patchlen)
        #     ]
        # )
        start_time = times[0]
        end_time = times[-1]
        def convert_to_epoch(yyyymmdd):
            time = str(yyyymmdd)
            year = int(time[:4])
            month = int(time[4:6])
            day = int(time[6:])
            # Use python library
            import datetime
            epoch = datetime.datetime(year, month, day).timestamp() // 86400
            return int(epoch)
        def convert_to_yyyymmdd(epoch):
            import datetime
            date = datetime.datetime.fromtimestamp(epoch * 86400)
            return int(date.strftime("%Y%m%d"))
        time_lists = []
        for time in range(convert_to_epoch(start_time), convert_to_epoch(end_time) + 1, time_window):
            time_list = []
            # Only append times that are in the dataset
            for day in range(time, time + time_window + 1):
                if convert_to_yyyymmdd(day) in times:
                    time_list.append(time_select(dataset, convert_to_yyyymmdd(day)))
            if len(time_list) > 0:
                time_lists.append(time_merge_edge_time(time_list))
        self.time_dataset = time_merge(time_lists)

        print("# time-dataset: ", len(self.time_dataset))

        print(
            f"""RedditTroll Dataset(T={len(times)},metadata={self.metadata},dataset={dataset},
            )"""
        )

    @property
    def test_dataset(self):
        data = [(self.time_dataset, self.eval_datas[2])]
        return data

    @property
    def val_dataset(self):
        data = [(self.time_dataset, self.eval_datas[1])]
        return data

    @property
    def train_dataset(self):
        data = [(self.time_dataset, self.eval_datas[0])]
        return data

    def to(self, device):
        self.device = device
        self.dataset = move_to(self.dataset, self.device)
        self.time_dataset = move_to(self.time_dataset, device)
        self.eval_datas = move_to(self.eval_datas, device)


if __name__ == "__main__":
    time_window = 5
    shuffle = True
    test_full = False
    is_dynamic = False
    dataset = RedditTrollUniDataset(
        time_window=time_window,
        shuffle=shuffle,
        test_full=test_full,
        is_dynamic=is_dynamic,
    )

    dataset.train_dataset
    dataset.dataset
    dataset.metadata
    dataset.eval_datas
    dataset.train_dataset
    dataset.val_dataset[1]
    dataset.train_dataset[0][1]

    dataset.dataset
