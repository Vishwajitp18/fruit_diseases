
import torch

from torch.utils.data import Dataset

from PIL import Image


class MangoDiseaseDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_transform=None,
        environmental_data=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_transform = image_transform

        self.environmental_data = environmental_data


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        # ==========================================
        # IMAGE
        # ==========================================

        image_path = self.dataframe.iloc[
            index
        ]["image_path"]

        image = Image.open(
            image_path
        ).convert("RGB")


        # Apply image transformation
        if self.image_transform is not None:

            image = self.image_transform(
                image
            )


        # ==========================================
        # LABEL
        # ==========================================

        label = int(
            self.dataframe.iloc[
                index
            ]["label"]
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )


        # ==========================================
        # ENVIRONMENT
        # ==========================================

        if self.environmental_data is not None:

            environment = self.environmental_data[
                index
            ]

            if not isinstance(
                environment,
                torch.Tensor
            ):

                environment = torch.tensor(
                    environment,
                    dtype=torch.float32
                )

            else:

                environment = environment.float()

        else:

            # Empty tensor for image-only models
            environment = torch.empty(
                0,
                dtype=torch.float32
            )


        # ==========================================
        # RETURN SAMPLE
        # ==========================================

        return {
            "image": image,
            "environment": environment,
            "label": label
        }
