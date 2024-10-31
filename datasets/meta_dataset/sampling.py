from typing import Union, List, Tuple
import numpy as np
from numpy.random import RandomState
from . import dataset_spec as dataset_spec_lib
from . import imagenet_specification
from . config import EpisodeDescriptionConfig
from .dataset_spec import HierarchicalDatasetSpecification as HDS
from .dataset_spec import BiLevelDatasetSpecification as BDS
from .dataset_spec import DatasetSpecification as DS
from .utils import Split



MAX_SPANNING_LEAVES_ELIGIBLE = 392


def sample_num_ways_uniformly(num_classes: int,
                              min_ways: int,
                              max_ways: int):
    """Samples a number of ways for an episode uniformly and at random.

    The support of the distribution is [min_ways, num_classes], or
    [min_ways, max_ways] if num_classes > max_ways.

    Args:
      num_classes: int, number of classes.
      min_ways: int, minimum number of ways.
      max_ways: int, maximum number of ways. Only used if num_classes > max_ways.

    Returns:
      num_ways: int, number of ways for the episode.
    """
    max_ways = min(max_ways, num_classes)
    return np.random.randint(low=min_ways, high=max_ways + 1)


def sample_class_ids_uniformly(num_ways: int,
                               rel_classes: List[int]):
    """Samples the (relative) class IDs for the episode.

    Args:
      num_ways: int, number of ways for the episode.
      rel_classes: list of int, available class IDs to sample from.

    Returns:
      class_ids: np.array, class IDs for the episode, with values in rel_classes.
    """
    return np.random.choice(rel_classes, num_ways, replace=False)


def compute_num_query(images_per_class: np.ndarray,
                      max_num_query: int,
                      num_support: Union[int, Tuple[int, int]]):
    """Computes the number of query examples per class in the episode.

    Query sets are balanced, i.e., contain the same number of examples for each
    class in the episode.

    The number of query examples satisfies the following conditions:
    - it is no greater than `max_num_query`
    - if support size is unspecified, it is at most half the size of the
      smallest class in the episode
    - if support size is specified, it is at most the size of the smallest class
      in the episode minus the max support size.

    Args:
      images_per_class: np.array, number of images for each class.
      max_num_query: int, number of images for each class.
      num_support: int or tuple(int, int), number (or range) of support
        images per class.

    Returns:
      num_query: int, number of query examples per class in the episode.
    """
    if not num_support:
        if images_per_class.min() < 2:
            raise ValueError('Expected at least 2 images per class.')
        return np.minimum(max_num_query, (images_per_class // 2).min())
    elif isinstance(num_support, int):
        max_support = num_support
    else:
        _, max_support = num_support
    if (images_per_class - max_support).min() < 1:
        raise ValueError(
            'Expected at least {} images per class'.format(max_support + 1))
    return np.minimum(max_num_query, images_per_class.min() - max_support)


def sample_support_set_size(num_remaining_per_class: np.ndarray,
                            max_support_size_contrib_per_class: int,
                            max_support_set_size: int):
    """Samples the size of the support set in the episode.

    That number is such that:

    * The contribution of each class to the number is no greater than
      `max_support_size_contrib_per_class`.
    * It is no greater than `max_support_set_size`.
    * The support set size is greater than or equal to the number of ways.

    Args:
      num_remaining_per_class: np.array, number of images available for each class
        after taking into account the number of query images.
      max_support_size_contrib_per_class: int, maximum contribution for any given
        class to the support set size. Note that this is not a limit on the number
        of examples of that class in the support set; this is a limit on its
        contribution to computing the support set _size_.
      max_support_set_size: int, maximum size of the support set.

    Returns:
      support_set_size: int, size of the support set in the episode.
    """
    if max_support_set_size < len(num_remaining_per_class):
        raise ValueError('max_support_set_size is too small to have at least one '
                         'support example per class.')
    beta = np.random.uniform()
    support_size_contributions = np.minimum(max_support_size_contrib_per_class,
                                            num_remaining_per_class)
    return np.minimum(




        np.floor(beta * support_size_contributions + 1).sum(),
        max_support_set_size)


def sample_num_support_per_class(images_per_class: np.ndarray,
                                 num_remaining_per_class: np.ndarray,
                                 support_set_size: int,
                                 min_log_weight: float,
                                 max_log_weight: float):
    """Samples the number of support examples per class.

    At a high level, we wish the composition to loosely match class frequencies.
    Sampling is done such that:

    * The number of support examples per class is no greater than
      `support_set_size`.
    * The number of support examples per class is no greater than the number of
      remaining examples per class after the query set has been taken into
      account.

    Args:
      images_per_class: np.array, number of images for each class.
      num_remaining_per_class: np.array, number of images available for each class
        after taking into account the number of query images.
      support_set_size: int, size of the support set in the episode.
      min_log_weight: float, minimum log-weight to give to any particular class.
      max_log_weight: float, maximum log-weight to give to any particular class.

    Returns:
      num_support_per_class: np.array, number of support examples for each class.
    """
    if support_set_size < len(num_remaining_per_class):
        raise ValueError('Requesting smaller support set than the number of ways.')
    if np.min(num_remaining_per_class) < 1:
        raise ValueError('Some classes have no remaining examples.')



    remaining_support_set_size = support_set_size - len(num_remaining_per_class)

    unnormalized_proportions = images_per_class * np.exp(
        np.random.uniform(min_log_weight, max_log_weight, size=images_per_class.shape))
    support_set_proportions = (
        unnormalized_proportions / unnormalized_proportions.sum())


    num_desired_per_class = np.floor(
        support_set_proportions * remaining_support_set_size).astype('int32') + 1

    return np.minimum(num_desired_per_class, num_remaining_per_class)


class EpisodeDescriptionSampler(object):
    """Generates descriptions of Episode composition.

    In particular, for each Episode, it will generate the class IDs (relative to
    the selected split of the dataset) to include, as well as the number of
    support and query examples for each class ID.
    """

    def __init__(self,
                 dataset_spec: Union[HDS, BDS, DS],
                 split: Split,
                 episode_descr_config: EpisodeDescriptionConfig,
                 use_dag_hierarchy: bool = False,
                 use_bilevel_hierarchy: bool = False,
                 use_all_classes: bool = False,
                 ignore_hierarchy_probability: float = 0.0):
        """Initializes an EpisodeDescriptionSampler.episode_config.

        Args:
          dataset_spec: DatasetSpecification, dataset specification.
          split: one of Split.TRAIN, Split.VALID, or Split.TEST.
          episode_descr_config: An instance of EpisodeDescriptionConfig containing
            parameters relating to sampling shots and ways for episodes.
          pool: A string ('train' or 'test') or None, indicating which example-level
            split to select, if the current dataset has them.
          use_dag_hierarchy: Boolean, defaults to False. If a DAG-structured
            ontology is defined in dataset_spec, use it to choose related classes.
          use_bilevel_hierarchy: Boolean, defaults to False. If a bi-level ontology
            is defined in dataset_spec, use it for sampling classes.
          use_all_classes: Boolean, defaults to False. Uses all available classes,
            in order, instead of sampling. Overrides `num_ways` to the number of
            classes in `split`.
          ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
            the sampler ignore the hierarchy for this proportion of episodes and
            instead sample categories uniformly.

        Raises:
          RuntimeError: if required parameters are missing.
          ValueError: Inconsistent parameters.
        """
        self.dataset_spec = dataset_spec
        self.split = split
        self.use_dag_hierarchy = use_dag_hierarchy
        self.use_bilevel_hierarchy = use_bilevel_hierarchy
        self.ignore_hierarchy_probability = ignore_hierarchy_probability
        self.use_all_classes = use_all_classes
        self.num_ways = episode_descr_config.num_ways
        self.num_support = episode_descr_config.num_support
        self.num_query = episode_descr_config.num_query
        self.min_ways = episode_descr_config.min_ways
        self.max_ways_upper_bound = episode_descr_config.max_ways_upper_bound
        self.max_num_query = episode_descr_config.max_num_query
        self.max_support_set_size = episode_descr_config.max_support_set_size
        self.max_support_size_contrib_per_class = episode_descr_config.max_support_size_contrib_per_class
        self.min_log_weight = episode_descr_config.min_log_weight
        self.max_log_weight = episode_descr_config.max_log_weight
        self.min_examples_in_class = episode_descr_config.min_examples_in_class

        self.class_set = dataset_spec.get_classes(self.split)
        self.num_classes = len(self.class_set)

        self._filtered_class_set = []

        skipped_classes = []
        for class_id in self.class_set:
            n_examples = dataset_spec.get_total_images_per_class(class_id)
            if n_examples < self.min_examples_in_class:
                skipped_classes.append((class_id, n_examples))
            else:
                self._filtered_class_set.append(class_id)
        self.num_filtered_classes = len(self._filtered_class_set)

        if skipped_classes:
            print(
                  'Skipping the following classes, which do not have at least '
                  '%d examples', self.min_examples_in_class)
        for class_id, n_examples in skipped_classes:
            print('%s (ID=%d, %d examples)',
                  dataset_spec.class_names[class_id], class_id, n_examples)

        if self.min_ways and self.num_filtered_classes < self.min_ways:
            raise ValueError(
                '"min_ways" is set to {}, but split {} of dataset {} only has {} '
                'classes with at least {} examples ({} total), so it is not possible '
                'to create an episode for it. This may have resulted from applying a '
                'restriction on this split of this dataset by specifying '
                'benchmark.restrict_classes or benchmark.min_examples_in_class.'
                .format(self.min_ways, split, dataset_spec.name,
                        self.num_filtered_classes, self.min_examples_in_class,
                        self.num_classes))

        if self.use_all_classes:
            if self.num_classes != self.num_filtered_classes:
                raise ValueError('"use_all_classes" is not compatible with a value of '
                                 '"min_examples_in_class" ({}) that results in some '
                                 'classes being excluded.'.format(
                                   self.min_examples_in_class))
            self.num_ways = self.num_classes


        if episode_descr_config.ignore_dag_ontology:
            self.use_dag_hierarchy = False
        if episode_descr_config.ignore_bilevel_ontology:
            self.use_bilevel_hierarchy = False


        if self.use_bilevel_hierarchy:
            print('=======================')
            print('Using bilevel hierarchy !')
            print('=======================')
            if self.num_ways:
                raise ValueError('"use_bilevel_hierarchy" is incompatible with '
                                 '"num_ways".')
            if self.min_examples_in_class > 0:
                raise ValueError('"use_bilevel_hierarchy" is incompatible with '
                                 '"min_examples_in_class".')
            if self.use_dag_hierarchy:
                raise ValueError('"use_bilevel_hierarchy" is incompatible with '
                                 '"use_dag_hierarchy".')

            if not isinstance(dataset_spec,
                              dataset_spec_lib.BiLevelDatasetSpecification):
                raise ValueError('Only applicable to datasets with a bi-level '
                                 'dataset specification.')

            all_superclasses = dataset_spec.get_superclasses(self.split)
            self.superclass_set = []
            for i in all_superclasses:
                if self.dataset_spec.classes_per_superclass[i] < self.min_ways:
                    raise ValueError(
                        'Superclass: %d has num_classes=%d < min_ways=%d.' %
                        (i, self.dataset_spec.classes_per_superclass[i], self.min_ways))
                self.superclass_set.append(i)

        elif self.use_dag_hierarchy:
            if self.num_ways:
                raise ValueError('"use_dag_hierarchy" is incompatible with "num_ways".')

            if not isinstance(dataset_spec,
                              dataset_spec_lib.HierarchicalDatasetSpecification):
                raise ValueError('Only applicable to datasets with a hierarchical '
                                 'dataset specification.')
            print('=======================')
            print('Using DAG hierarchy !')
            print('=======================')


            graph = dataset_spec.get_split_subgraph(self.split)



            class_set = self.class_set
            abs_to_rel_ids = dict((abs_id, i) for i, abs_id in enumerate(class_set))


            leaves = set(imagenet_specification.get_leaves(graph))
            internal_nodes = graph - leaves


            spanning_leaves_dict = imagenet_specification.get_spanning_leaves(graph)



            self.span_leaves_rel = []
            for node in internal_nodes:
                node_leaves = spanning_leaves_dict[node]


                ids_rel = []
                for leaf in node_leaves:
                    abs_id = dataset_spec.class_names_to_ids[leaf.wn_id]
                    if abs_id in self._filtered_class_set:
                        ids_rel.append(abs_to_rel_ids[abs_id])



                if self.min_ways <= len(ids_rel) <= MAX_SPANNING_LEAVES_ELIGIBLE:
                    self.span_leaves_rel.append(ids_rel)
            self.span_leaves_rel.sort(key=lambda l: sum(l))
            num_eligible_nodes = len(self.span_leaves_rel)
            self.span_leaves_rel = np.array(self.span_leaves_rel, dtype=object)
            if num_eligible_nodes < 1:
                raise ValueError('There are no classes eligible for participating in '
                                 'episodes. Consider changing the value of '
                                 '`EpisodeDescriptionSampler.min_ways` in gin, or '
                                 'or MAX_SPANNING_LEAVES_ELIGIBLE in data.py.')

    def sample_class_ids(self):
        """Returns the (relative) class IDs for an episode.

        If self.use_dag_hierarchy, it samples them according to a procedure
        informed by the dataset's ontology, otherwise randomly.
        If self.min_examples_in_class > 0, classes with too few examples will not
        be selected.
        """
        prob = [1.0, 0.0]
        if self.ignore_hierarchy_probability:
          prob = [
              1.0 - self.ignore_hierarchy_probability,
              self.ignore_hierarchy_probability
          ]

        if self.use_dag_hierarchy and np.random.choice([True, False], p=prob):


            episode_classes_rel = np.random.choice(self.span_leaves_rel)


            if len(episode_classes_rel) > self.max_ways_upper_bound:
                episode_classes_rel = np.random.choice(
                    episode_classes_rel,
                    size=[self.max_ways_upper_bound],
                    replace=False)


            assert len(episode_classes_rel) >= self.min_ways
            assert len(episode_classes_rel) <= self.max_ways_upper_bound
        elif self.use_bilevel_hierarchy and np.random.choice([True, False], p=prob):



            episode_superclass = np.random.choice(self.superclass_set, 1)[0]
            num_superclass_classes = self.dataset_spec.classes_per_superclass[
                episode_superclass]

            num_ways = sample_num_ways_uniformly(
                num_superclass_classes,
                min_ways=self.min_ways,
                max_ways=self.max_ways_upper_bound)






            episode_subclass_ids = sample_class_ids_uniformly(num_ways,
                                                              num_superclass_classes)
            (episode_classes_rel,
             _) = self.dataset_spec.get_class_ids_from_superclass_subclass_inds(
                 self.split, episode_superclass, episode_subclass_ids)
        elif self.use_all_classes:
            episode_classes_rel = np.arange(self.num_classes)
        else:
            if self.num_ways:
                num_ways = self.num_ways
            else:
                num_ways = sample_num_ways_uniformly(
                    self.num_filtered_classes,
                    min_ways=self.min_ways,
                    max_ways=self.max_ways_upper_bound)

            ids_rel = [
                class_id - self.class_set[0] for class_id in self._filtered_class_set
            ]
            episode_classes_rel = sample_class_ids_uniformly(num_ways, ids_rel)

        return episode_classes_rel

    def sample_episode_description(self):
        """Returns the composition of an episode.

        Returns:
          A sequence of `(class_id, num_support, num_query)` tuples, where
            relative `class_id` is an integer in [0, self.num_classes).
        """
        class_ids = self.sample_class_ids()
        images_per_class = np.array([
            self.dataset_spec.get_total_images_per_class(
                self.class_set[cid]) for cid in class_ids
        ])

        if self.num_query:
            num_query = self.num_query
        else:
            num_query = compute_num_query(
                images_per_class,
                max_num_query=self.max_num_query,
                num_support=self.num_support)

        if self.num_support:
            if isinstance(self.num_support, int):
                if any(self.num_support + num_query > images_per_class):
                    raise ValueError('Some classes do not have enough examples.')
                num_support = self.num_support
            else:
                start, end = self.num_support
                if any(end + num_query > images_per_class):
                    raise ValueError('The range provided for uniform sampling of the '  
                                     'number of support examples per class is not valid: '
                                     'some classes do not have enough examples.')
                num_support = np.random.randint(low=start, high=end + 1)
            num_support_per_class = [num_support for _ in class_ids]
        else:
            num_remaining_per_class = images_per_class - num_query
            support_set_size = sample_support_set_size(
                num_remaining_per_class,
                self.max_support_size_contrib_per_class,
                max_support_set_size=self.max_support_set_size)
            num_support_per_class = sample_num_support_per_class(
                images_per_class,
                num_remaining_per_class,
                support_set_size,
                min_log_weight=self.min_log_weight,
                max_log_weight=self.max_log_weight)

        return tuple(
            (class_id, num_support, num_query)
            for class_id, num_support in zip(class_ids, num_support_per_class))

    def compute_chunk_sizes(self):
        """Computes the maximal sizes for the flush, support, and query chunks.

        Sequences of dataset IDs are padded with dummy IDs to make sure they can be
        batched into episodes of equal sizes.

        The "flush" part of the sequence has a size that is upper-bounded by the
        size of the "support" and "query" parts.

        If variable, the size of the "support" part is in the worst case

            max_support_set_size,

        and the size of the "query" part is in the worst case

            max_ways_upper_bound * max_num_query.

        Returns:
          The sizes of the flush, support, and query chunks.
        """
        if not self.num_ways:
            max_num_ways = self.max_ways_upper_bound
        else:
            max_num_ways = self.num_ways

        if not self.num_support:
            support_chunk_size = self.max_support_set_size
        elif isinstance(self.num_support, int):
            support_chunk_size = max_num_ways * self.num_support
        else:
            largest_num_support_per_class = self.num_support[1]
            support_chunk_size = max_num_ways * largest_num_support_per_class

        if not self.num_query:
            max_num_query = self.max_num_query
        else:
            max_num_query = self.num_query
        query_chunk_size = max_num_ways * max_num_query

        flush_chunk_size = support_chunk_size + query_chunk_size
        return (flush_chunk_size, support_chunk_size, query_chunk_size)
