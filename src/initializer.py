import hashlib, json, os, random, yaml, thop, warnings, logging, pynvml, torch, numpy as np
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import utils as U
from . import dataset
from . import model
from . import scheduler


class Initializer():
    def __init__(self, args):
        self.args = args
        self.init_save_dir()

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_experiment_identity()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_save_dir(self):
        self.save_dir = U.set_logging(self.args)
        with open('{}/config.yaml'.format(self.save_dir), 'w') as f:
            yaml.dump(vars(self.args), f)
        logging.info('Saving folder path: {}'.format(self.save_dir))

    def init_environment(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            pynvml.nvmlInit()
            for i in self.args.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memused = meminfo.used / 1024 / 1024
                logging.info('GPU-{} used: {}MB'.format(i, memused))
                if memused > 8000:
                    pynvml.nvmlShutdown()
                    logging.info('')
                    logging.error('GPU-{} is occupied!'.format(i))
                    raise ValueError()
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        dataset_args['debug'] = self.args.debug
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.dataset, **dataset_args
        )
        self.train_loader = DataLoader(self.feeders['train'],
            batch_size=self.train_batch_size, num_workers=4*len(self.args.gpus),
                                       pin_memory=True, shuffle=True, drop_last=True,
                                       persistent_workers=False
        )
        self.eval_loader = DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, num_workers=4*len(self.args.gpus),
                                      pin_memory=True, shuffle=False, drop_last=False,
                                      persistent_workers=False
        )
        self.location_loader = self.feeders['location'] if dataset_name == 'ntu' else None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))
        self._save_counterfactual_profile_artifact()

    def _save_counterfactual_profile_artifact(self):
        feeder = self.feeders['train']
        if not (getattr(feeder, 'counterfactual_pair', False) or getattr(feeder, 'neighbor_mode', 'none') == 'swap'):
            return

        artifact_path = os.path.join(
            self.save_dir, 'counterfactual_train_profiles.npz'
        )
        temporary_path = artifact_path + '.tmp.npz'
        try:
            np.savez_compressed(
                temporary_path,
                subject_ids=feeder.counterfactual_profile_subject_ids,
                bone_lengths=feeder.counterfactual_profiles,
                capture_samples_used=feeder.counterfactual_profile_counts,
                capture_subject_ids=(
                    feeder.counterfactual_profile_capture_subject_ids
                ),
                capture_names=feeder.counterfactual_profile_capture_names,
                connect_joint=np.asarray(feeder.conn, dtype=np.int64),
                profile_sha256=np.asarray(
                    feeder.counterfactual_profile_sha256
                ),
                source=np.asarray('NTU training split only'),
                profile_seed=np.asarray(
                    feeder.counterfactual_seed, dtype=np.int64
                ),
                max_action=np.asarray(
                    feeder.counterfactual_max_action, dtype=np.int64
                ),
            )
            os.replace(temporary_path, artifact_path)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise
        logging.info(
            'Counterfactual train-only profile artifact: {} (sha256-{})'.format(
                artifact_path, feeder.counterfactual_profile_sha256
            )
        )

    @staticmethod
    def _portable_protocol_value(value):
        """Drop workstation-only paths while retaining scientific settings."""
        excluded_path_keys = {
            'root_folder', 'ntu60_path', 'ntu120_path',
            'canonical_bone_path',
        }
        if isinstance(value, dict):
            return {
                str(key): Initializer._portable_protocol_value(item)
                for key, item in sorted(value.items())
                if key not in excluded_path_keys
            }
        if isinstance(value, (list, tuple)):
            return [
                Initializer._portable_protocol_value(item) for item in value
            ]
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _feeder_manifest_sha256(feeder):
        digest = hashlib.sha256()
        for name, label, seq_len in zip(
            feeder.name, feeder.label, feeder.seq_len
        ):
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            basename = str(name).replace('\\', '/').rsplit('/', 1)[-1]
            digest.update(basename.encode('utf-8'))
            digest.update(b'\0')
            digest.update(np.asarray(
                [int(label), int(seq_len)], dtype='<i8'
            ).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _feeder_coordinate_sample_sha256(feeder, sample_count=257):
        """Hash deterministic full-coordinate rows without scanning a 10GB file."""
        data = feeder.data
        digest = hashlib.sha256()
        digest.update(str(data.dtype).encode('ascii'))
        digest.update(np.asarray(data.shape, dtype='<i8').tobytes())
        count = min(int(sample_count), len(feeder))
        indices = np.unique(np.linspace(
            0, len(feeder) - 1, num=count, dtype=np.int64
        ))
        for index in indices:
            digest.update(np.asarray(index, dtype='<i8').tobytes())
            digest.update(
                np.ascontiguousarray(data[int(index)]).tobytes()
            )
        return digest.hexdigest(), int(len(indices))

    @staticmethod
    def _normalized_source_sha256(path):
        with open(path, 'r', encoding='utf-8') as handle:
            source = handle.read().replace('\r\n', '\n').replace('\r', '\n')
        return hashlib.sha256(source.encode('utf-8')).hexdigest()

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def init_experiment_identity(self):
        """Bind resume checkpoints to data, code and training protocol."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        source_paths = {
            'main.py': os.path.join(project_root, 'main.py'),
            'src/initializer.py': __file__,
            'src/processor.py': os.path.join(project_root, 'src', 'processor.py'),
            'src/utils.py': os.path.join(project_root, 'src', 'utils.py'),
            'src/dataset/ntu_feeder.py': os.path.join(
                project_root, 'src', 'dataset', 'ntu_feeder.py'
            ),
            'src/dataset/__init__.py': os.path.join(
                project_root, 'src', 'dataset', '__init__.py'
            ),
            'src/dataset/graphs.py': os.path.join(
                project_root, 'src', 'dataset', 'graphs.py'
            ),
            'src/model/__init__.py': os.path.join(
                project_root, 'src', 'model', '__init__.py'
            ),
            'src/model/activations.py': os.path.join(
                project_root, 'src', 'model', 'activations.py'
            ),
            'src/model/attentions.py': os.path.join(
                project_root, 'src', 'model', 'attentions.py'
            ),
            'src/model/fusions.py': os.path.join(
                project_root, 'src', 'model', 'fusions.py'
            ),
            'src/model/nets.py': os.path.join(
                project_root, 'src', 'model', 'nets.py'
            ),
            'src/model/layers.py': os.path.join(
                project_root, 'src', 'model', 'layers.py'
            ),
            'src/model/geometry_consistency.py': os.path.join(
                project_root, 'src', 'model', 'geometry_consistency.py'
            ),
            'src/model/person_invariance.py': os.path.join(
                project_root, 'src', 'model', 'person_invariance.py'
            ),
            'src/scheduler/__init__.py': os.path.join(
                project_root, 'src', 'scheduler', '__init__.py'
            ),
            'src/scheduler/lr_schedulers.py': os.path.join(
                project_root, 'src', 'scheduler', 'lr_schedulers.py'
            ),
        }
        source_sha256 = {
            name: self._normalized_source_sha256(path)
            for name, path in source_paths.items()
        }
        train_feeder = self.feeders['train']
        eval_feeder = self.feeders['eval']
        train_coordinate_sha256, train_coordinate_samples = (
            self._feeder_coordinate_sample_sha256(train_feeder)
        )
        eval_coordinate_sha256, eval_coordinate_samples = (
            self._feeder_coordinate_sample_sha256(eval_feeder)
        )
        protocol = {
            'version': 1,
            'config_id': self.args.config,
            'seed': int(self.args.seed),
            'debug': bool(self.args.debug),
            'dataset': self.args.dataset,
            'dataset_args': self._portable_protocol_value(
                self.args.dataset_args
            ),
            'model_type': self.args.model_type,
            'model_args': self._portable_protocol_value(self.args.model_args),
            'training_args': self._portable_protocol_value(
                getattr(self.args, 'training_args', {}) or {}
            ),
            'optimizer': self.args.optimizer,
            'optimizer_args': self._portable_protocol_value(
                self.args.optimizer_args
            ),
            'lr_scheduler': self.args.lr_scheduler,
            'scheduler_args': self._portable_protocol_value(
                self.args.scheduler_args
            ),
            'data_shape': [int(value) for value in self.data_shape],
            'num_class': int(self.num_class),
            'train_sample_count': len(train_feeder),
            'eval_sample_count': len(eval_feeder),
            'train_manifest_sha256': self._feeder_manifest_sha256(
                train_feeder
            ),
            'eval_manifest_sha256': self._feeder_manifest_sha256(eval_feeder),
            'train_data_file_size': os.path.getsize(train_feeder.data_path),
            'eval_data_file_size': os.path.getsize(eval_feeder.data_path),
            'train_coordinate_sample_count': train_coordinate_samples,
            'eval_coordinate_sample_count': eval_coordinate_samples,
            'train_coordinate_sample_sha256': train_coordinate_sha256,
            'eval_coordinate_sample_sha256': eval_coordinate_sha256,
            'counterfactual_profile_sha256': getattr(
                train_feeder, 'counterfactual_profile_sha256', ''
            ),
            'train_batch_size': int(self.train_batch_size),
            'micro_batches_per_epoch': self.train_micro_batches_per_epoch,
            'gradient_accumulation_steps': (
                self.gradient_accumulation_steps
            ),
            'steps_per_epoch': self.optimizer_steps_per_epoch,
            'source_sha256': source_sha256,
        }
        if getattr(train_feeder, 'neighbor_mode', 'none') != 'none':
            protocol['neighbor_calibration_sha256'] = train_feeder.neighbor_calibration_sha256
        protocol_json = json.dumps(
            protocol, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True,
        )
        self.experiment_fingerprint = hashlib.sha256(
            protocol_json.encode('utf-8')
        ).hexdigest()

        initial_checkpoint_sha256 = ''
        if (
            not self.args.resume
            and self.args.pretrained_checkpoint
            and os.path.isfile(self.args.pretrained_checkpoint)
        ):
            initial_checkpoint_sha256 = self._file_sha256(
                self.args.pretrained_checkpoint
            )
        self.experiment_metadata = {
            'fingerprint': self.experiment_fingerprint,
            'protocol': protocol,
            'runtime': {
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'configured_gpu_count': len(self.args.gpus),
                'train_num_workers': self.train_loader.num_workers,
                'initial_checkpoint': self.args.pretrained_checkpoint,
                'initial_checkpoint_sha256': initial_checkpoint_sha256,
            },
        }

        self._write_experiment_protocol()
        logging.info(
            'Experiment fingerprint: {}'.format(
                self.experiment_fingerprint
            )
        )

    def _write_experiment_protocol(self):
        protocol_path = os.path.join(self.save_dir, 'experiment_protocol.json')
        temporary_path = protocol_path + '.tmp'
        try:
            with open(temporary_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    self.experiment_metadata, handle, indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, protocol_path)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise

    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,
            'num_class': self.num_class,
            'A': torch.Tensor(self.A),
            'parts': self.parts,
        }
        self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs)
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        with open('{}/model.txt'.format(self.save_dir), 'w') as f:
            print(self.model, file=f)
        flops, params = thop.profile(deepcopy(self.model), inputs=torch.rand([1,1]+self.data_shape), verbose=False)
        logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))
        self.model = torch.nn.DataParallel(
            self.model.to(self.device), device_ids=self.args.gpus, output_device=self.output_device
        )
        if (
            not self.args.resume
            and self.args.pretrained_checkpoint
            and self.args.pretrained_path
        ):
            raise ValueError(
                'Use either pretrained_checkpoint or pretrained_path, not both'
            )
        pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)
        if self.args.resume:
            # Processor.start() restores the self-contained experiment
            # checkpoint after optimizer/scheduler construction.  Do not make
            # resume depend on the original initialization file still being at
            # the same workstation-specific path.
            logging.info(
                'Resume requested: pretrained initialization is skipped.'
            )
        elif self.args.pretrained_checkpoint:
            if not os.path.isfile(self.args.pretrained_checkpoint):
                raise FileNotFoundError(
                    'Pretrained checkpoint does not exist: {}'.format(
                        self.args.pretrained_checkpoint
                    )
                )
            checkpoint = torch.load(
                self.args.pretrained_checkpoint,
                map_location=torch.device('cpu'),
                weights_only=False,
            )
            self.model.module.load_state_dict(checkpoint['model'], strict=True)
            if 'best_state' in checkpoint:
                self.cm = checkpoint['best_state'].get('cm', 0)
            logging.info(
                'Pretrained checkpoint: {}'.format(
                    self.args.pretrained_checkpoint
                )
            )
        elif os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'), weights_only=False)
            self.model.module.load_state_dict(checkpoint['model'])
            self.cm = checkpoint['best_state']['cm']
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}!'.format(pretrained_model))
            logging.info('Create model randomly.')

    def init_optimizer(self):
        try:
            optimizer = U.import_class('torch.optim.{}'.format(self.args.optimizer))
        except:
            logging.warning('Warning: Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = U.import_class('torch.optim.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        training_args = getattr(self.args, 'training_args', {}) or {}
        self.gradient_accumulation_steps = int(
            training_args.get('gradient_accumulation_steps', 1)
        )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError('gradient_accumulation_steps must be positive')
        self.optimizer_steps_per_epoch = (
            len(self.train_loader) // self.gradient_accumulation_steps
        )
        if self.optimizer_steps_per_epoch <= 0:
            raise ValueError(
                'gradient_accumulation_steps exceeds the training loader length'
            )
        self.train_micro_batches_per_epoch = (
            self.optimizer_steps_per_epoch
            * self.gradient_accumulation_steps
        )
        skipped_micro_batches = (
            len(self.train_loader) - self.train_micro_batches_per_epoch
        )
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(
            self.args.lr_scheduler,
            self.optimizer_steps_per_epoch,
            **scheduler_args
        )
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))
        logging.info(
            'Gradient accumulation: {} micro-batch(es)/step, {} optimizer '
            'steps/epoch, {} trailing micro-batch(es) skipped'.format(
                self.gradient_accumulation_steps,
                self.optimizer_steps_per_epoch,
                skipped_micro_batches,
            )
        )

    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))
