import os
import signal

from absl import app
from absl import flags
from absl import logging
import docker
from docker import types

flags.DEFINE_bool(
    'use_gpu', True, 'whether run with GPUs'
)
flags.DEFINE_string(
    'gpu_device', 'all', 'which GPUs to use'
)
flags.DEFINE_string(
    'output_dir', '/tmp/CLEAN', 'absolute directory to save outputs and trained models'
)
flags.DEFINE_string(
    'docker_image_name', 'clean_train', 'name of the docker image'
)
flags.DEFINE_float(
    'learning_rate', 5e-4, 'learning rate'
)
flags.DEFINE_integer(
    'epoch', 2000, 'number of epochs'
)
flags.DEFINE_string(
    'model_name', 'split10_triplet', 'name of the model'
)
flags.DEFINE_string(
    'training_data', 'split10', 'name of the training data'
)
flags.DEFINE_integer(
    'hidden_dim', 512, 'hidden dimension'
)
flags.DEFINE_integer(
    'out_dim', 128, 'output dimension'
)
flags.DEFINE_integer(
    'adaptive_rate', 100, 'adaptive rate'
)
flags.DEFINE_bool(
    'verbose', False, 'whether print verbose'
)
FLAGS = flags.FLAGS

def main(argv):

    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    mounts = []
    command_args = []

    if not os.path.exists(os.path.join(FLAGS.output_dir, 'model')):
        os.makedirs(os.path.join(FLAGS.output_dir, 'model'))
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'esm_data')):
        os.makedirs(os.path.join(FLAGS.output_dir, 'esm_data'))
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'distance_map')):
        os.makedirs(os.path.join(FLAGS.output_dir, 'distance_map'))

    volumes = {
        f'{os.path.join(FLAGS.output_dir, "model")}': {'bind': '/app/CLEAN/data/model', 'mode': 'rw'},
        f'{os.path.join(FLAGS.output_dir, "esm_data")}': {'bind': '/app/CLEAN/data/esm_data', 'mode': 'rw'},
        f'{os.path.join(FLAGS.output_dir, "distance_map")}': {'bind': '/app/CLEAN/data/distance_map', 'mode': 'rw'},
        f'{os.path.join(os.path.expanduser("~"), ".cache/torch/hub/checkpoints")}': {'bind': '/root/.cache/torch/hub/checkpoints', 'mode': 'rw'},
    }

    command_args.extend([
        f'-l={FLAGS.learning_rate}',
        f'-e={FLAGS.epoch}',
        f'-n={FLAGS.model_name}',
        f'-t={FLAGS.training_data}',
        f'-d={FLAGS.hidden_dim}',
        f'-o={FLAGS.out_dim}',
        f'--adaptive_rate={FLAGS.adaptive_rate}',
        f'--verbose={FLAGS.verbose}',
    ])

    client = docker.from_env()
    device_requests = [
        docker.types.DeviceRequest(driver='nvidia', capabilities=[['gpu']])
    ] if FLAGS.use_gpu else None

    container = client.containers.run(
        image=FLAGS.docker_image_name,
        command=command_args,
        device_requests=device_requests,
        remove=True,
        detach=True,
        volumes=volumes,
        environment={
            'NVIDIA_VISIBLE_DEVICES': FLAGS.gpu_device,
        }
    )

    signal.signal(signal.SIGINT, lambda sig, frame: container.kill())

    for line in container.logs(stream=True):
        logging.info(line.strip().decode('utf-8'))
    
if __name__ == '__main__':
    app.run(main)