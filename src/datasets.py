#!/usr/bin/env python

import os
import natsort
import torch
from glob import glob
from typing import Callable
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from multiprocessing import Manager
from torchvision.transforms import Compose, Resize, InterpolationMode, Lambda
from torchvision.io import read_image
from src.transforms import (
    normalize,
    CropDict,
    RandomRotateFlipDict,
)

class AidDataset(Dataset):
    """
    lr, hr (단일) 데이터셋 클래스
    """
    def __init__(
        self,
        dataset_path: str,
        transform: Compose=None,
        use_cache: bool=True,
        multiprocessing_manager=None,
        **kws,
    ):
        
        self.dataset_path = dataset_path
        self.transform = transform or Compose([])
        # Avoid forcing multiprocessing.Manager in notebook/single-process runs to prevent pickling issues
        self.multiprocessing_manager = multiprocessing_manager
        self.paths = self.load_and_natsort_img_paths(dataset_path)
        self.use_cache = use_cache
        # Use a regular dict to cache tensors to avoid multiprocessing pickling issues
        self.cache = {} # self.multiprocessing_manager.dict()

    def __len__(self):
        """
        Return:
            데이셋 크기
        """
        return len(self.paths)

    def __getitem__(self, idx: int):
        """
        - 이미지가 이전에 가져왔다면 캐시에서 반환
        - 그렇지 않다면 디스크에서 가져와서 변환(첫 에포크 로딩 시 느리지만 이후로는 캐싱 사용)

        Args:
            idx (int): 이미지 인덱스

        Returns:
            Tensor: 텐서로 변환된 이미지
        """
        if idx not in self.cache:
            path = self.paths[idx]

            img = read_image(path)
            img = self.transform(img)

            self.cache_img(idx, img)

            return img

        else:
            # 이미 캐시에 있는 경우 캐시에서 반환
            return self.cache[idx]

    def load_and_natsort_img_paths(self, dataset_path: str):
        """
        이미지 경로 로드 및 자연 정렬을 수행하는 함수

        Args:
            dataset_path (str): 데이터셋 경로

        Return:
            multiprocessing.Manager.list: 자연 정렬된 이미지 경로 목록(멀티프로세스 사용 시 공유 메모리 사용)
        """
        paths = glob(os.path.join(dataset_path, "*.png"))

        sorted_paths = natsort.natsorted(paths)
        if self.multiprocessing_manager is not None:
            return self.multiprocessing_manager.list(sorted_paths)
        return sorted_paths

    def cache_img(self, idx, x):
        """
        이미지를 캐시에 저장하는 함수

        Args:
            idx (int): 이미지 인덱스
            x (Tensor): 저장할 이미지
        """
        if self.use_cache:
            self.cache[idx] = x

class TransformDataset(Dataset):
    """
    PyTorch Dataset 클래스를 사용하여 이미지를 가져올 때 변환을 적용하는 클래스
    Reference: https://gist.github.com/alkalait/c99213c164df691b5e37cd96d5ab9ab2#file-sn7dataset-py-L278
    """
    def __init__(self, dataset: Dataset, transform: Callable) -> None:
        """
        PyTorch Dataset 클래스를 사용하여 이미지를 가져올 때 변환을 적용하는 클래스

        Args:
            dataset (torch.Dataset): 변환을 적용할 데이터셋
            transform (Callable): 적용할 변환
        """
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int):
        """
        데이터셋에서 이미지를 가져오고 변환을 적용하는 함수

        Params:
            idx (int): 가져올 이미지의 인덱스

        Return:
            Tensor: 변환이 적용된 이미지
        """
        img = self.dataset[idx]
        return self.transform(img)

    def __len__(self):
        return len(self.dataset)


class DictDataset(Dataset):
    """
    PyTorch Dataset 클래스를 사용하여 데이터셋 딕셔너리를 랩핑하는 클래스
    - get 메서드에 전달된 인덱스는 단일 AOI의 이미지 또는 크롭(크롭 적용 시)을 참조
    - get 메서드는 각 데이터셋에 대한 이미지 인덱스의 이미지 딕셔너리를 반환
    """

    def __init__(self, **dictionary_of_datasets: dict[str, Dataset]):
        """
        Args:
            dictionary_of_datasets (dict): 랩핑할 torch 데이터셋 딕셔너리
        """
        self.datasets = {
            dataset_type: dataset
            for dataset_type, dataset in dictionary_of_datasets.items()
            if isinstance(dataset, Dataset)
        }

    def __getitem__(self, idx):
        """
        데이터셋에서 이미지를 가져오는 함수

        Params:
            idx (int): 가져올 이미지의 인덱스

        Return:
            dict: 각 데이터셋에서 가져온 이미지
        """
        return {dataset_type: dataset[idx] for dataset_type, dataset in self.datasets.items()}

    def __len__(self):
        """
        랩핑된 데이터셋 딕셔너리에서 가장 작은 데이터셋의 길이를 반환하는 함수

        Return:
            int: 랩핑된 데이터셋 딕셔너리에서 가장 작은 데이터셋의 길이
        """
        return min(len(dictionary) for dictionary in self.datasets.values())


def load_aid_dataset(**kws) -> dict[str, DataLoader]:
    """
    Args:
        root: 데이터셋 루트 경로(default: AID-dataset/)
        zoom_factor (int): 업스케일 인자 (default: 4 | 2, 3, 4) 
        output_size (tuple[int, int]): HR 리사이즈 크기 (default: (600, 600))
        chip_size (tuple[int, int]): HR 기준 칩 크기 (default: (600, 600))
        chip_stride (tuple[int, int]): HR 기준 칩 스트라이드 (default: (600, 600))
        batch_size (int): 배치 크기(default: 8)
        batch_size_test (int): 테스트 배치 크기(default: batch_size)
        normalize_lr (bool): LR 정규화 여부(default: True)
        normalize_hr (bool): HR 정규화 여부(default: True)
        shuffle (bool): 데이터셋 셔플 여부(default: True)
        interpolation (InterpolationMode): LR, HR 리사이즈 방법 (default: InterpolationMode.BICUBIC)
        randomly_rotate_and_flip_images: 데이터 회전 및 뒤집기 여부(default: False)
        data_split_seed (int): 데이터셋 셔플 시드(default: 42)
        subset_train (float): 트레이닝 데이터셋 비율(default: None | 0.0 ~ 1.0)
        num_workers (int): 데이터로더에 사용할 서브프로세스 수(default: 0)

    Returns:
        dict[str, DataLoader]: 데이터로더(train, val, test)
        (e.g., {'train': DataLoader, 'val': DataLoader, 'test': DataLoader})
    """
    kws.setdefault("zoom_factor", 4)
    kws.setdefault("output_size", (600, 600))
    kws.setdefault("input_size", (kws["output_size"][0] // kws["zoom_factor"], kws["output_size"][1] // kws["zoom_factor"]))
    kws.setdefault("chip_size", kws["output_size"])
    kws.setdefault("chip_stride", kws["chip_size"])
    kws.setdefault("batch_size", 8)
    kws.setdefault("batch_size_test", kws["batch_size"])

    kws.setdefault("normalize_lr", True)
    kws.setdefault("normalize_hr", True)
    kws.setdefault("shuffle", True)
    kws.setdefault("interpolation", InterpolationMode.BICUBIC)
    kws.setdefault("randomly_rotate_and_flip_images", False)

    kws.setdefault("root", "AID-dataset/")
    kws.setdefault("train_split", None)
    kws.setdefault("val_split", None)
    kws.setdefault("test_split", None)
    kws.setdefault("data_split_seed", 42)
    kws.setdefault("subset_train", None)

    kws.setdefault("num_workers", 0)

    lr_dir_postfix = f"_x{kws['zoom_factor']}"
    transforms = make_transforms(**kws)

    return make_dataloaders(
        transforms=transforms,
        sub_dir={
            "hr": "HR",
            "lr": f"LR{lr_dir_postfix}"
        },
        **kws,
    )


def make_dataloaders(
    root: str,
    sub_dir: dict[str, str],
    output_size: tuple[int, int],
    chip_size: tuple[int, int],
    chip_stride: tuple[int, int],
    transforms: dict[str, Compose],
    shuffle: bool = True,
    data_split_seed: int=42,
    subset_train=None,
    randomly_rotate_and_flip_images=True,
    **kws,
) -> dict[str, DataLoader]:

    multiprocessing_manager = Manager()
    datasets_arguments = generate_dataset_arguments_from_kws(
        root,
        sub_dir,
        transforms,
        multiprocessing_manager,
    )

    datasets = generate_datasets(datasets_arguments) # dict[str, DictDataset] (e.g., {'train': <DictDataset>, 'val': <DictDataset>, 'test': <DictDataset>})

    if shuffle:
        datasets = shuffle_datasets(datasets, data_split_seed)
    datasets, number_of_chips = generate_chipped_and_augmented_datasets(
        datasets,
        chip_size,
        chip_stride,
        output_size,
        randomly_rotate_and_flip_images
    ) # 변환을 적용한 ConcatDataset(DictDataset -> ConcatDataset)

    if type(datasets) is dict:
        dataset_train, dataset_val, dataset_test = (
            datasets["train"][0], # 'train': (train_dataset_object, train_chip_count)
            datasets["val"][0],
            datasets["test"][0],
        )

    if subset_train is not None:
        dataset_train = reduce_training_set(dataset_train, subset_train)

    test_dataloader, train_dataloader, val_dataloader = create_dataloaders_for_datasets(
        dataset_test, dataset_train, dataset_val, kws
    )

    print(f"Train set size: {len(dataset_train)}")
    print(f"Val set size: {len(dataset_val)}")
    print(f"Test set size: {len(dataset_test)}")

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def generate_dataset_arguments_from_kws(
    root: str,
    sub_dir: dict[str, str],
    transforms: dict[str, Compose],
    multiprocessing_manager: Manager,
) -> dict[str, dict]:

    return {
        img_type: dict(
            root=root,
            sub_dir=sub_dir[img_type],
            transform=transforms[img_type],
            multiprocessing_manager=multiprocessing_manager,
        )
        for img_type in ["lr", "hr"]
    }


def generate_datasets(datasets_arguments: dict[str, dict]) -> dict[str, DictDataset]: # (e.g., {'train': <DictDataset>, 'val': <DictDataset>, 'test': <DictDataset>})

    return {
        split: generate_datasets_for_split(
            split,
            datasets_arguments
        )
        for split in ["train", "val", "test"]
    }


def generate_datasets_for_split(split: str, datasets_arguments: dict[str, dict]):

    datasets = {}
    for dataset_type, arguments in datasets_arguments.items():
        dataset_path = os.path.join(arguments["root"], split, arguments["sub_dir"])
        arguments["dataset_path"] = dataset_path
        print(f"{split}_{dataset_type} dataset path: {dataset_path}")
        datasets[dataset_type] = AidDataset(**arguments) # lr, hr 따로

    dataset_dict = DictDataset(**datasets) # lr, hr 묶어서

    return dataset_dict


def shuffle_datasets(datasets, data_split_seed: int):
    """
    train, val, test 데이터셋 각각을 랜덤하게 섞는 함수

    returns:
        Subset: 섞인 데이터셋(인덱스 순서 변경)
    """
    print(f"Shuffling the dataset splits using {data_split_seed}")

    if isinstance(datasets, dict):
        return {
            split: shuffle_datasets(dataset, data_split_seed)
            for split, dataset in datasets.items()
        }
    number_of_scenes = len(datasets)
    (datasets,) = random_split(
        datasets,
        [number_of_scenes,],
        generator=torch.Generator().manual_seed(data_split_seed),
    )
    return datasets


def generate_chipped_and_augmented_datasets(
    datasets,
    chip_size,
    chip_stride,
    output_size,
    randomly_rotate_and_flip_images,
) -> tuple[dict[str, DictDataset], None]:

    if isinstance(datasets, dict):
        return (
            {
                key: generate_chipped_and_augmented_datasets(
                    value,
                    chip_size,
                    chip_stride,
                    output_size,
                    randomly_rotate_and_flip_images,
                )
                for key, value in datasets.items()
            },
            None,
        )
    number_of_scenes = len(datasets)

    dataset, number_of_chips = apply_chipped_and_augmentation_to_datasets(
        datasets,
        output_size,
        chip_size,
        chip_stride,
        randomly_rotate_and_flip_images,
    )

    dataset = transpose_scenes_and_chips(dataset, number_of_chips, number_of_scenes) # 칩 순서대로 쌓인 데이터셋을 씬 순서대로 쌓인 데이터셋으로 변환
    return dataset, number_of_chips

def apply_chipped_and_augmentation_to_datasets(
    dataset_dict,
    output_size,
    chip_size,
    chip_stride,
    randomly_rotate_and_flip_images,
):
    """
    데이터셋을 크롭 및 회전 및 뒤집기 변환을 적용하는 함수

    Params:
        dataset_dict (DictDataset): 크롭 및 회전할 데이터셋
        output_size (int): 출력 이미지 크기
        chip_size (int): 칩/패치 크기
        chip_stride (int): 칩/패치 스트라이드
        randomly_rotate_and_flip_images (bool): 이미지 랜덤 회전 및 뒤집기 여부

    Return:
        Tuple[ConcatDataset, int]: 크롭 및 회전된 데이터셋, 칩 수
    """

    dataset_dict_grid = []
    output_height, output_width = output_size
    chip_height, chip_width = chip_size
    stride_height, stride_width = chip_stride

    # Make sure chip isn't larger than the output size
    assert chip_height <= output_height and chip_width <= output_width

    last_stride_step_x = output_width - chip_width + 1
    last_stride_step_y = output_height - chip_height + 1
    for stride_step_x in range(0, last_stride_step_x, stride_width):
        for stride_step_y in range(0, last_stride_step_y, stride_height):
            transform_dict = Compose(
                [
                    CropDict(
                        stride_step_x, stride_step_y, chip_width, chip_height, src="hr"
                    ),
                    RandomRotateFlipDict(angles=[0, 90, 180, 270]) if randomly_rotate_and_flip_images else Compose([]),
                ]
            )
            dataset_dict_grid.append(
                TransformDataset(dataset_dict, transform=transform_dict)
            )
    dataset = torch.utils.data.ConcatDataset(dataset_dict_grid) # 데이터셋 여러 개를 합침(ConcatDataset)
    return dataset, len(dataset_dict_grid)


def transpose_scenes_and_chips(dataset, number_of_chips, number_of_scenes):
    """
    데이터셋 샘플을 재정렬하여 서로 다른 장면의 칩들이 교차되도록하는 함수

    데이터셋이 여러 개의 칩을 포함하는 장면들로 구성되어 있다고 가정
    이 함수는 인덱스를 재배열하여 다음과 같은 순서의 데이터셋을 생성:
    [chip0_scene0, chip0_scene1, ..., chip0_sceneN, chip1_scene0, chip1_scene1, ..., chip1_sceneN, ...]

    Params:
        dataset (torch.utils.data.Dataset): 장면 단위로 순차적으로 구성된 원본 데이터셋
        number_of_scenes (int): 데이터셋의 고유한 장면 수
        number_of_chips (int): 장면당 칩의 수

    Return:
        torch.utils.data.Subset: 샘플이 재정렬된 원본 데이터셋의 Subset
    """
    # Transpose scenes and chips
    # indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., number_of_scenes * number_of_chips]
    indices = Tensor(range(number_of_scenes * number_of_chips)).int()
    # indices = [0, number_of_scenes, 2*number_of_scenes, ..., 1+number_of_scenes, 2+2*number_of_scenes, ... ]
    transposed_indices = indices.reshape(number_of_chips, number_of_scenes).T.reshape(
        indices.numel()
    )
    dataset = torch.utils.data.Subset(dataset, transposed_indices)
    assert len(dataset) == number_of_scenes * number_of_chips
    return dataset


def reduce_training_set(dataset_train, subset_train):
    """
    훈련 데이터셋을 주어진 비율로 줄이는 함수

    Params:
        dataset_train (Dataset): 훈련 데이터셋
        subset_train (float): 훈련 데이터셋 사용 비율

    Return:
        Dataset: 줄여진 훈련 데이터셋
    """
    # Reduce the train set if needed
    if subset_train < 1:
        dataset_train = torch.utils.data.Subset(
            dataset_train, list(range(int(subset_train * len(dataset_train))))
        )
    return dataset_train


def create_dataloaders_for_datasets(dataset_test, dataset_train, dataset_val, kws):
    """
    데이터셋에서 PyTorch 데이터로더를 생성하는 함수

    Params:
        dataset_test (torch.utils.data.Dataset): 테스트 데이터셋
        dataset_train (torch.utils.data.Dataset): 훈련 데이터셋
        dataset_val (torch.utils.data.Dataset): 검증 데이터셋
        kws (dict): 데이터로더 생성에 사용될 키워드 인자

    Return:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader: 테스트, 훈련 및 검증 데이터셋에 대한 데이터로더
    """
    batch_size, batch_size_test, number_of_workers = kws.get("batch_size", 1), kws.get("batch_size_test", 1), kws.get("num_workers", 1)
    train_dataloader = DataLoader(
        dataset_train,
        num_workers=number_of_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True if number_of_workers > 0 else False,
    )
    # val_loader = DataLoader(dataset_val, num_workers=W, batch_size=len(dataset_val), pin_memory=True,
    val_dataloader = DataLoader(
        dataset_val,
        num_workers=number_of_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True if number_of_workers > 0 else False,
    )
    # test_loader = DataLoader(dataset_test, num_workers=W, batch_size=len(dataset_test), pin_memory=True)
    test_dataloader = DataLoader(
        dataset_test,
        num_workers=number_of_workers,
        batch_size=batch_size_test,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True if number_of_workers > 0 else False,
    )
    return test_dataloader, train_dataloader, val_dataloader
    
def make_transforms(
    input_size,
    output_size,
    interpolation,
    normalize_lr,
    normalize_hr,
    **kws,
):

    transforms = {}
    # lr에 대한 변환 함수
    transforms["lr"] = Compose(
        [
            Lambda(normalize) if normalize_lr else Compose([]),
            Resize(size=input_size, interpolation=interpolation, antialias=True),
        ]
    )
 
    # hr에 대한 변환 함수
    transforms["hr"] = Compose(
        [
            Lambda(normalize) if normalize_hr else Compose([]),
            Resize(size=output_size, interpolation=interpolation, antialias=True),
        ]
    )

    return transforms
