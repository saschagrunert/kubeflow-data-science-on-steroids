#!/usr/bin/env python3

import json
import os.path as path
from typing import Dict

from kfp.compiler import Compiler
from kfp.dsl import ContainerOp, ExitHandler, pipeline
from kubernetes import client as k8s

OUT_DIR = '/out'
METADATA_FILE = 'mlpipeline-ui-metadata.json'
METRICS_FILE = 'mlpipeline-metrics.json'
METADATA_FILE_PATH = path.join(OUT_DIR, METADATA_FILE)
METRICS_FILE_PATH = path.join(OUT_DIR, METRICS_FILE)


@pipeline(name='My pipeline', description='')
def pipeline():

    deploy = demo_op('deploy', is_exit_handler=True)
    with ExitHandler(deploy):
        deps = demo_op('setup dependencies')

        analyze = demo_op('analyze data')
        analyze.after(deps)

        train1 = demo_op('training 1')
        train2 = demo_op('training 2')
        train3 = demo_op('training 3')
        train1.after(analyze)
        train2.after(analyze)
        train3.after(analyze)

        predict = demo_op('predict')
        predict.after(train1)
        predict.after(train2)
        predict.after(train3)

        matrix = demo_op('create confusion-matrix')
        roc = demo_op('create roc')
        matrix.after(predict)
        roc.after(predict)


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


def demo_op(name: str, is_exit_handler=False) -> ContainerOp:
    op = ContainerOp(name=name,
                     image='alpine:latest',
                     command=['sh', '-c'],
                     arguments=[
                         'echo "Running step $0" && echo "$1" > $2',
                         name,
                         markdown_metadata(name),
                         METADATA_FILE_PATH,
                     ],
                     is_exit_handler=is_exit_handler,
                     output_artifact_paths=default_artifact_path())
    op.add_volume(
        k8s.V1Volume(name='volume',
                     empty_dir=k8s.V1EmptyDirVolumeSource())).add_volume_mount(
                         k8s.V1VolumeMount(name='volume', mount_path=OUT_DIR))
    return op


def default_artifact_path() -> Dict[str, str]:
    return {
        path.splitext(METADATA_FILE)[0]: METADATA_FILE_PATH,
        path.splitext(METRICS_FILE)[0]: METRICS_FILE_PATH,
    }
