from .crossdomain import CrossDomainUniDataset
from .ecomm import EcommUniDataset
from .yelp import YelpNCLFDataset
from .reddittroll import RedditTrollUniDataset


def load_data(args):
    model, shuffle, test_full, is_dynamic = (
        args.model,
        args.shuffle,
        args.test_full,
        args.dynamic,
    )
    if args.dataset == "Aminer":
        time_window = args.twin = (
            13 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = CrossDomainUniDataset(
            time_window=time_window,
            shuffle=shuffle,
            test_full=test_full,
            is_dynamic=is_dynamic,
        )
        args.in_dim, args.hid_dim, args.out_dim = (
            -1 if not args.homo else 32,
            8,
            8,
        )  # -1 refers to lazy parameterization for hetero models
        args.predict_type = "author"

    elif args.dataset == "Ecomm":
        time_window = args.twin = (
            7 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = EcommUniDataset(
            time_window=time_window,
            shuffle=shuffle,
            test_full=test_full,
            is_dynamic=is_dynamic,
        )
        args.in_dim, args.hid_dim, args.out_dim = (
            -1 if not args.homo else 8,
            8,
            8,
        )  # -1 refers to lazy parameterization for hetero models
        args.predict_type = ["user", "item"]

    elif args.dataset == "Yelp-nc":
        time_window = args.twin = (
            12 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = YelpNCLFDataset(
            time_window=time_window,
            shuffle=shuffle,
            test_full=test_full,
            is_dynamic=is_dynamic,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        args.in_dim, args.hid_dim, args.out_dim = (
            -1 if not args.homo else 32,
            8,
            8,
        )  # -1 refers to lazy parameterization for hetero models
        args.num_classes = 3
        args.predict_type = "item"
    elif args.dataset == "Reddit-troll":
        time_window = args.twin = (
            180 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = RedditTrollUniDataset(
            time_window=time_window,
            shuffle=shuffle,
            test_full=test_full,
            is_dynamic=is_dynamic,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        if args.twin >= 1:
            args.twin = len(dataset.time_dataset)
        args.in_dim = -1 if not args.homo else 770
        # args.in_dim, args.hid_dim, args.out_dim = (
        #     -1 if not args.homo else 770,
        #     32,
        #     32,
        # )  # -1 refers to lazy parameterization for hetero models
        args.num_classes = 1
        args.predict_type = "author"
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
    return dataset, args
