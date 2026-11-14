import os, sys, shutil, logging, json, random, torch, numpy as np
from time import time, strftime, localtime


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_time(total_time):
    s = int(total_time % 60)
    m = int(total_time / 60) % 60
    h = int(total_time / 60 / 60) % 24
    d = int(total_time / 60 / 60 / 24)
    return '{:0>2d}d-{:0>2d}h-{:0>2d}m-{:0>2d}s'.format(d, h, m, s)


def get_current_timestamp():
    ct = time()
    ms = int((ct - int(ct)) * 1000)
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)


def load_checkpoint(work_dir, model_name='resume'):
    if model_name == 'resume':
        file_name = '{}/checkpoint.pth.tar'.format(work_dir)
    elif model_name == 'debug':
        file_name = '{}/temp/debug.pth.tar'.format(work_dir)
    else:
        dirs, accs = {}, {}
        work_dir = '{}/{}'.format(work_dir, model_name)
        if os.path.exists(work_dir):
            for i, dir_time in enumerate(os.listdir(work_dir)):
                if os.path.isdir('{}/{}'.format(work_dir, dir_time)):
                    state_file = '{}/{}/reco_results.json'.format(work_dir, dir_time)
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            best_state = json.load(f)
                        accs[str(i+1)] = best_state['acc_top1']
                        dirs[str(i+1)] = dir_time
        if len(dirs) == 0:
            logging.warning('Warning: Do NOT exists any model in workdir!')
            logging.info('Evaluating initial or pretrained model.')
            return None
        logging.info('Please choose the evaluating model from the following models.')
        logging.info('Default is the initial or pretrained model.')
        for key in dirs.keys():
            logging.info('({}) accuracy: {:.2%} | training time: {}'.format(key, accs[key], dirs[key]))
        logging.info('Your choice (number of the model, q for quit): ')
        while True:
            idx = input(get_current_timestamp())
            if idx == '':
                logging.info('Evaluating initial or pretrained model.')
                return None
            elif idx in dirs.keys():
                break
            elif idx == 'q':
                logging.info('Quit!')
                sys.exit(1)
            else:
                logging.info('Wrong choice!')
        file_name = '{}/{}/{}.pth.tar'.format(work_dir, dirs[idx], model_name)
    try:
        checkpoint = torch.load(file_name, map_location=torch.device('cpu'), weights_only=False)
    except:
        logging.info('')
        logging.error('Error: Wrong in loading this checkpoint: {}!'.format(file_name))
        raise ValueError()
    return checkpoint


def capture_rng_state():
    """Capture all RNGs used by the training and data pipelines."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(checkpoint):
    """Restore RNGs from a new-format checkpoint, while accepting old ones."""
    rng_state = checkpoint.get('rng_state')
    if rng_state is None:
        logging.warning(
            'Warning: This is a legacy checkpoint without RNG state; '
            'resume is supported, but its random stream cannot be reproduced exactly.'
        )
        return False

    random.setstate(rng_state['python'])
    np.random.set_state(rng_state['numpy'])
    torch.set_rng_state(rng_state['torch'])

    cuda_states = rng_state.get('cuda')
    if cuda_states is not None and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(cuda_states) != device_count:
            logging.warning(
                'Warning: Checkpoint contains RNG states for {} CUDA devices, '
                'but {} devices are currently visible; restoring matching devices only.'.format(
                    len(cuda_states), device_count
                )
            )
        for device_idx, state in enumerate(cuda_states[:device_count]):
            torch.cuda.set_rng_state(state, device=device_idx)
    return True


def validate_resume_checkpoint(
    checkpoint, model_name, expected_fingerprint=None
):
    """Prevent a new-format checkpoint from being resumed by another experiment."""
    checkpoint_model_name = checkpoint.get('model_name')
    if checkpoint_model_name is None:
        logging.warning(
            'Warning: Legacy checkpoint has no experiment identity; '
            'please verify that -c and work_dir match the original run.'
        )
        return
    if checkpoint_model_name != model_name:
        raise ValueError(
            'Checkpoint experiment mismatch: expected {}, found {}. '
            'Do not use -r across different configs or work directories.'.format(
                model_name, checkpoint_model_name
            )
        )
    if expected_fingerprint is None:
        return
    saved_fingerprint = checkpoint.get('experiment_fingerprint')
    if saved_fingerprint is None:
        logging.warning(
            'Warning: Legacy checkpoint has no experiment fingerprint; '
            'data/code/protocol compatibility cannot be verified.'
        )
        return
    if saved_fingerprint != expected_fingerprint:
        raise ValueError(
            'Checkpoint experiment fingerprint mismatch: expected {}, '
            'found {}. Do not resume after changing data, code, batch size, '
            'counterfactual protocol, optimizer, or scheduler.'.format(
                expected_fingerprint, saved_fingerprint
            )
        )


def save_checkpoint(
    model, optimizer, scheduler, epoch, best_state, is_best,
    work_dir, save_dir, model_name, experiment_fingerprint=None,
    experiment_metadata=None,
):
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {
        'model':model, 'optimizer':optimizer, 'scheduler':scheduler,
        'best_state':best_state, 'epoch':epoch,
        'checkpoint_version':3, 'model_name':model_name,
        'experiment_fingerprint':experiment_fingerprint,
        'experiment_metadata':experiment_metadata,
        'rng_state':capture_rng_state(),
    }
    cp_name = '{}/checkpoint.pth.tar'.format(work_dir)
    temp_cp_name = '{}.tmp'.format(cp_name)
    try:
        with open(temp_cp_name, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_cp_name, cp_name)
    except:
        if os.path.exists(temp_cp_name):
            os.remove(temp_cp_name)
        raise
    if is_best:
        best_name = '{}/{}.pth.tar'.format(save_dir, model_name)
        temp_best_name = '{}.tmp'.format(best_name)
        shutil.copy(cp_name, temp_best_name)
        os.replace(temp_best_name, best_name)

        result_name = '{}/reco_results.json'.format(save_dir)
        temp_result_name = '{}.tmp'.format(result_name)
        result_state = {key:value for key, value in best_state.items() if key != 'cm'}
        with open(temp_result_name, 'w') as f:
            json.dump(result_state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_result_name, result_name)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def set_logging(args):
    if args.debug or args.evaluate or args.extract or args.visualize or args.generate_data:
        save_dir = '{}/temp'.format(args.work_dir)
    else:
        ct = strftime('%Y-%m-%d %H-%M-%S')
        save_dir = '{}/{}_{}_{}/{}'.format(args.work_dir, args.config, args.model_type, args.dataset, ct)
    create_folder(save_dir)
    log_format = '[ %(asctime)s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    handler = logging.FileHandler('{}/log.txt'.format(save_dir), mode='w', encoding='UTF-8')
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)
    return save_dir


class CrossEntropyLabelSmooth(torch.nn.Module):
	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss
