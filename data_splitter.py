from src.utils import folder_data_splitter_non_iid, folder_data_splitter_iid
import argparse

"""
Splits a folder of images that follow YOLOv5 guidelines into partitions to try a federated
approach to train a model

You have to choose:
    1 - which type of partition you want (iid, non_iid)
    2 - number of total partitions (agents involved)
    3 - id of the particular agent where the data will be stored
    4 - a folder to make the partition from 
"""


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["iid", "non_iid"],
        help="kind of partition to make",
    )
    parser.add_argument(
        "--folder_origin",
        type=str,
        required=True,
        default="/Users/joaquimmorera/Desktop/fedlearning_yolo/data/datasets/camacuc2",
        help="folder to make the partitions from",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        required=True,
        default=3,
        help="partitions to make in the dataset",
    )
    parser.add_argument(
        "--agent",
        type=int,
        required=True,
        default=1,
        help="Number of one of the partitions done in n_splits",
    )
    parser.add_argument(
        "--ext",
        type=str,
        help="suffix to at the name of the cloned folder",
    )

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_opt()
    print(type(opt.ext))

    if opt.type == "iid":

        if opt.ext == None:
            opt.ext = "_iid"

        folder_data_splitter_iid(
            folder_origin=opt.folder_origin,
            n_splits=opt.n_splits,
            agent=opt.agent,
            ext=opt.ext,
        )
    else:
        if opt.ext == None:
            opt.ext = "_non_iid"

        folder_data_splitter_non_iid(
            folder_origin=opt.folder_origin,
            n_splits=opt.n_splits,
            agent=opt.agent,
            ext=opt.ext,
        )
