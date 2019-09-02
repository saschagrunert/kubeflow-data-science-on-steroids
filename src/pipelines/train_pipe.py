#!/usr/bin/env python3

import json
import os.path as path
import sys
from functools import wraps
from typing import Dict

import kfp.dsl as dsl
import numpy as np
from kfp import compiler
from kfp.compiler import Compiler
from kfp.components import func_to_container_op as py_convert
from kfp.dsl import ContainerOp, ExitHandler, pipeline
from kubernetes import client as k8s

sys.path.append('python')  # isort:skip
from train import *  # isort:skip noqa

OUT_DIR = '/something'
METADATA_FILE = 'mlpipeline-ui-metadata.json'
METRICS_FILE = 'mlpipeline-metrics.json'
METADATA_FILE_PATH = path.join(OUT_DIR, METADATA_FILE)
METRICS_FILE_PATH = path.join(OUT_DIR, METRICS_FILE)
SQUEEZE_FILE = "squeezenet.model"
RES18_FILE = "res18.model"
RESNET_FILE = "resnet.model"
SQUEEZE_FILE_PATH = path.join(OUT_DIR, SQUEEZE_FILE)
RES18_FILE_PATH = path.join(OUT_DIR, RES18_FILE)
RESNET_FILE_PATH = path.join(OUT_DIR, RESNET_FILE)
DEPLOYED_MODEL = SQUEEZE_FILE

#### GCS definitions
BASE_IMAGE = 'mbu93/kubeflow:latest'
####

def storage_op(func, *args):
    op = py_convert(func, base_image=BASE_IMAGE)(*args)
    op.add_volume(k8s.V1Volume(name='volume',
                               host_path=k8s.V1HostPathVolumeSource(path='/data/out')))\
      .add_volume_mount(k8s.V1VolumeMount(name='volume', mount_path=OUT_DIR))
    return op


@pipeline(name='FastAI Training Pipeline', description='Train a set\
                                                        of various pytorch models\
                                                        using fastai. Get their accuracy\
                                                        and confusion matrices afterwards.')
def pipeline(pretrained: bool = False):
    deploy = demo_op('deploy', is_exit_handler=True)
    with ExitHandler(deploy):
        # setup the data loader
        setup_data = py_convert(setup_data_loader, base_image=BASE_IMAGE)()

        # fit all networks
        fit_squeeze = storage_op(fit_squeezenet, setup_data.output, OUT_DIR, pretrained)
        fit_res18 = storage_op(fit_resnet18, setup_data.output, OUT_DIR, pretrained)
        fit_res = storage_op(fit_resnet, setup_data.output, OUT_DIR, pretrained)
        fit_squeeze.after(setup_data)
        fit_res18.after(setup_data)
        fit_res.after(setup_data)
        fit_squeeze.default_artifact_paths = {path.splitext(SQUEEZE_FILE)[0]: SQUEEZE_FILE_PATH}
        fit_res18.default_artifact_paths = {path.splitext(RES18_FILE)[0]: RES18_FILE_PATH}
        fit_res.default_artifact_paths = {path.splitext(RESNET_FILE)[0]: RESNET_FILE_PATH}

        # get the network accuracys
        squeeze_acc = storage_op(get_accuracy, fit_squeeze.output)
        res18_acc = storage_op(get_accuracy, fit_res18.output)
        res_acc = storage_op(get_accuracy, fit_res.output)
        squeeze_acc.after(fit_squeeze)
        res18_acc.after(fit_res18)
        res_acc.after(fit_res)

        # get the confusion matrices
        squeeze_confusion = storage_op(get_confusion, fit_squeeze.output)
        res18_confusion = storage_op(get_confusion, fit_res18.output)
        res_confusion = storage_op(get_confusion, fit_res.output)
        squeeze_confusion.after(fit_squeeze)
        res18_confusion.after(fit_res18)
        res_confusion.after(fit_res)

        # save the best model for later deployment
        models = np.array([SQUEEZE_FILE_PATH, RES18_FILE_PATH, RESNET_FILE_PATH])
        best = np.argmax([squeeze_acc.output, res18_acc.output, res_acc.output])
        globals()['DEPLOYED_MODEL'] = models[best]

if __name__ == '__main__':
    Compiler().compile(pipeline)

def markdown_metadata(result: str) -> str:
    return json.dumps({
        'outputs': [{
            'type': 'markdown',
            'source': 'The result: %s' % result,
            'storage': 'inline',
        }]
    })

def demo_op(name: str, metadata=markdown_metadata,
            is_exit_handler=False) -> ContainerOp:
    op = ContainerOp(name=name,
                     image=BASE_IMAGE,
                     command=['sh', '-c'],
                     arguments=[
                         'echo "Running step $0" && echo "$1" > $2',
                         name,
                         metadata(name),
                         METADATA_FILE_PATH,
                     ],
                     is_exit_handler=is_exit_handler,
                     output_artifact_paths=default_artifact_path())
    op.add_volume(
        k8s.V1Volume(name='volume',
                     host_path=k8s.V1HostPathVolumeSource(path='/data/out')))\
        .add_volume_mount(k8s.V1VolumeMount(name='volume', mount_path=OUT_DIR))
    return op

def default_artifact_path() -> Dict[str, str]:
    return {
        path.splitext(METADATA_FILE)[0]: METADATA_FILE_PATH,
        path.splitext(METRICS_FILE)[0]: METRICS_FILE_PATH,
    }
