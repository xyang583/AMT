import os
import random
import h5py
from PIL import Image
import json

import numpy as np

import torch
from .meta_dataset import config as config_lib
from .meta_dataset import sampling
from .meta_dataset.utils import Split
from .meta_dataset.transform import get_transforms
from .meta_dataset import dataset_spec as dataset_spec_lib


class FullMetaDatasetH5(torch.utils.data.Dataset):
    def __init__(self, args, split=Split['TRAIN']):
        super().__init__()


        data_config = config_lib.DataConfig(args)
        episod_config = config_lib.EpisodeDescriptionConfig(args)

        self.need_global_label = False

        if split == Split.TRAIN:
            datasets = args.base_sources
            episod_config.num_episodes = args.nEpisode
            if args.need_global_label:
                self.need_global_label = True
        elif split == Split.VALID:
            datasets = args.val_sources
            episod_config.num_episodes = args.nValEpisode
        else:
            datasets = args.test_sources
            episod_config.num_episodes = args.nTestEpisode

        use_dag_ontology_list = [False]*len(datasets)
        use_bilevel_ontology_list = [False]*len(datasets)
        if episod_config.num_ways:
            if len(datasets) > 1:
                raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
        else:

            if 'omniglot' in datasets:
                use_bilevel_ontology_list[datasets.index('omniglot')] = True
            if 'ilsvrc_2012' in datasets:
                use_dag_ontology_list[datasets.index('ilsvrc_2012')] = True

        episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
        episod_config.use_dag_ontology_list = use_dag_ontology_list


        all_dataset_specs = []
        for dataset_name in datasets:
            dataset_records_path = os.path.join(data_config.path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            all_dataset_specs.append(dataset_spec)

        num_classes = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])
        print(f"=> There are {num_classes} classes in the {split} split of the combined datasets")

        self.datasets = datasets
        self.transforms = get_transforms(data_config, split, args)
        self.len = episod_config.num_episodes * len(datasets)

        self.class_map = {}
        self.class_h5_dict = {}
        self.class_samplers = {}
        self.class_images = {}

        for i, dataset_name in enumerate(datasets):
            dataset_spec = all_dataset_specs[i]
            base_path = dataset_spec.path
            class_set = dataset_spec.get_classes(split)
            num_classes = len(class_set)

            record_file_pattern = dataset_spec.file_pattern
            assert record_file_pattern.startswith('{}'), f'Unsupported {record_file_pattern}.'

            self.class_map[dataset_name] = {}
            self.class_h5_dict[dataset_name] = {}
            self.class_images[dataset_name] = {}

            for class_id in class_set:
                data_path = os.path.join(base_path, record_file_pattern.format(class_id))
                self.class_map[dataset_name][class_id] = data_path.replace('tfrecords', 'h5')
                self.class_h5_dict[dataset_name][class_id] = None
                self.class_images[dataset_name][class_id] = [str(j) for j in range(dataset_spec.get_total_images_per_class(class_id))]

            self.class_samplers[dataset_name] = sampling.EpisodeDescriptionSampler(
                dataset_spec=dataset_spec,
                split=split,
                episode_descr_config=episod_config,
                use_dag_hierarchy=episod_config.use_dag_ontology_list[i],
                use_bilevel_hierarchy=episod_config.use_bilevel_ontology_list[i],
                ignore_hierarchy_probability=args.ignore_hierarchy_probability)

    def __len__(self):
        return self.len

    def get_next(self, source, class_id, idx):
        h5_path = self.class_map[source][class_id]

        if self.class_h5_dict[source][class_id] is None:
            self.class_h5_dict[source][class_id] = h5py.File(h5_path, 'r')

        h5_file = self.class_h5_dict[source][class_id]
        record = h5_file[idx]
        x = record['image'][()]

        if self.transforms:
            x = Image.fromarray(x)
            x = self.transforms(x)

        return x

    def __getitem__(self, idx):
        support_images = []
        support_labels = []
        support_global_labels = []
        query_images = []
        query_labels = []
        query_global_labels = []
        source = np.random.choice(self.datasets)
        sampler = self.class_samplers[source]
        episode_description = sampler.sample_episode_description()
        episode_description = tuple(
            (class_id + sampler.class_set[0], num_support, num_query)
            for class_id, num_support, num_query in episode_description)

        episode_classes = list(class_ for class_, _, _ in episode_description)
        assert len(episode_classes) == len(set(episode_classes)), "redundant classes"

        for class_id, nb_support, nb_query in episode_description:
            assert nb_support + nb_query <= len(self.class_images[source][class_id]), \
                f'Failed fetching {nb_support + nb_query} images from {source} at class {class_id}.'
            random.shuffle(self.class_images[source][class_id])
            for j in range(0, nb_support):
                x = self.get_next(source, class_id, self.class_images[source][class_id][j])
                support_images.append(x)

            for j in range(nb_support, nb_support + nb_query):
                x = self.get_next(source, class_id, self.class_images[source][class_id][j])
                query_images.append(x)

            support_labels.extend([episode_classes.index(class_id)] * nb_support)
            support_global_labels.extend([class_id] * nb_support)
            query_labels.extend([episode_classes.index(class_id)] * nb_query)
            query_global_labels.extend([class_id] * nb_query)

        support_images = torch.stack(support_images, dim=0)
        query_images = torch.stack(query_images, dim=0)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)
        return support_images, support_labels, query_images, query_labels