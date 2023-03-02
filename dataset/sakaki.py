from collections import OrderedDict
import os

from dataset.base_dataset import BaseFiveClassTargetDataset

SAKAKI_CLASS_LIST = ["plant", "vegetation,", "artificial", "ground", "sky", "background"]

color_encoding = OrderedDict(
    [
        ("plant", (0, 255, 255)),
        ("terrain", (152, 251, 152)),
        ("artificial_objects", (255, 0, 0)),
        ("ground", (255, 255, 0)),
        ("sky", (70, 130, 180)),
        ("background", (0, 0, 0)),
    ]
)

color_palette = [
    0, 255, 255, 
    152, 251, 152,
    255, 0, 0, 
    255, 255, 0, 
    70, 130, 180,
    0, 0, 0
]


class SakakiDataset(BaseFiveClassTargetDataset):
    def __init__(
        self,
        list_name,
        label_root="",
        mode="train",
        size=(256, 480),
        is_hard_label=False,
        load_labels=True,
        max_iter=None,
        is_three_class=False,
    ):
        """Initialize a dataset

        Each line of a data list must be formatted as follows:
            rgb_image_path, depth_image_path, label_path, trav_mask_path, start_point_x, start_point_y, end_point_x, end_point_y

        Parameters
        ----------
        list_name: `str`
            File name of the data list
        root: `str`
            Name of the directory where the data list is stored
        train: `bool`
            True if the dataset is for training
        size: `list`
            Image size to which the image is resized
        raw: `bool`
            True if the original pixel coordinates are used for training of the path line.
          Otherwise, they are scaled in [0, 1]
        rough_plant: `bool`
            Use rough annotation
        is_soft_label: `bool`
            `True` if soft labels are used
        load_labels: `bool`
            Whether to load label data.
        is_old_label: `bool`
            If `True`, consider the labels as from an old label set
            i.e., ['traversable plant', 'other plant', 'artificial object', 'ground'].
            Otherwise, ['plant', 'artificial object', 'ground']

        """
        super().__init__(
            label_root=label_root,
            mode=mode,
            size=size,
            is_hard_label=is_hard_label,
            load_labels=load_labels,
            max_iter=max_iter,
            is_three_class=is_three_class,
        )

        self.data_file = list_name

        # Initialize the lists
        with open(self.data_file, "r") as lines:
            for line in lines:
                # Split a line
                line_split = line.split(",")

                #
                # RGB
                #
                rgb_img_loc = line_split[0].rstrip()
                # Chieck the first character. If it's '%', the line is not read
                if rgb_img_loc == "" or rgb_img_loc[0] == "%":
                    continue
                # Verify the existence of the file
                if rgb_img_loc != "" and not os.path.isfile(rgb_img_loc):
                    print("Not found : " + rgb_img_loc)
                    assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)

                #
                # Segmentation label
                #
                if self.load_labels:
                    if self.label_root:
                        label_img_loc = os.path.join(self.label_root, line_split[1].rstrip())
                    else:
                        label_img_loc = line_split[1].rstrip()

                    # if self.mode == "train" and not self.is_hard_label:
                    if not self.is_hard_label:
                        label_img_loc = label_img_loc.replace("png", "pt")
                else:
                    label_img_loc = ""

                # Verify the existence of the file
                if (
                    self.load_labels
                    and label_img_loc != ""
                    and not os.path.isfile(label_img_loc)
                ):
                    print("Not found : " + label_img_loc)
                    assert os.path.isfile(label_img_loc)
                self.labels.append(label_img_loc)

        if self.max_iter is not None and self.max_iter > len(self.images):
            self.images *= (self.max_iter // len(self.images))
            self.labels *= (self.max_iter // len(self.labels))
