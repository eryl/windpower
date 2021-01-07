import unittest
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import time

from mltrain.hyperparameter import HyperParameterManager, HyperParameterTrainer, DiscreteHyperParameter, gather_hyper_prams

class TestHyperParameterMaterialization(unittest.TestCase):
    def test_materialize_hyper_parameters(self):
        base_args = ('foo', 2, ('a', 'b', 'c'), DiscreteHyperParameter([3, 5]))
        base_kwargs = {'baz': 'bar', 'asdf': DiscreteHyperParameter([True, False])}

        for i in range(5):
            print(materialize_hyper_parameters(dict(base_args=base_args,
                                                    base_kwargs=base_kwargs)))

        hp_obj = HyperParameterTest(base_args=base_args, base_kwargs=base_kwargs, extra=TestModel())
        for i in range(10):
            print(materialize_hyper_parameters(hp_obj))

    def test_gather_hyper_params(self):
        base_args = ('foo', 2, ('a', 'b', 'c'), DiscreteHyperParameter([3, 5, {'a': DiscreteHyperParameter([True, False])}]))
        base_kwargs = {'baz': 'bar', 'asdf': DiscreteHyperParameter([True, False]), 'args': base_args}
        hp_list = []
        param_obj = gather_hyper_prams(base_kwargs, hp_list)
        print(param_obj)
