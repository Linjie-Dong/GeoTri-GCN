import logging, os, re, torch, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer
from .model.geometry_consistency import (
    linear_consistency_scale,
    symmetric_js_divergence,
    symmetric_stop_gradient_cosine,
)


class Processor(Initializer):

    _ntu_person_pattern = re.compile(r'P(\d{3})(?=R\d{3}A\d{3})')

    @classmethod
    def _parse_ntu_person_id(cls, name):
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        sample_name = os.path.basename(str(name))
        match = cls._ntu_person_pattern.search(sample_name)
        if match is None:
            raise ValueError('Cannot parse NTU person ID from sample name: {}'.format(name))
        return int(match.group(1))

    def _init_person_adversarial(self):
        if hasattr(self, 'person_adversarial'):
            return

        model_args = self.args.model_args
        self.person_adversarial = bool(model_args.get('person_adversarial', False))
        if not self.person_adversarial:
            return

        self.grl_max = float(model_args.get('grl_max', 0.1))
        self.grl_gamma = float(model_args.get('grl_gamma', 10.0))
        self.person_loss_weight = float(model_args.get('person_loss_weight', 1.0))
        if self.grl_max < 0.0:
            raise ValueError('grl_max must be non-negative')
        if self.grl_gamma <= 0.0:
            raise ValueError('grl_gamma must be positive')
        if self.person_loss_weight <= 0.0:
            raise ValueError('person_loss_weight must be positive')

        parsed_person_ids = [
            self._parse_ntu_person_id(name)
            for name in self.feeders['train'].name
        ]
        person_ids = sorted(set(parsed_person_ids))
        self.num_person_class = int(model_args.get('num_person_class', 0))
        if not self.args.debug and len(person_ids) != self.num_person_class:
            raise ValueError(
                'Training data contains {} person IDs, but num_person_class is {}'.format(
                    len(person_ids), self.num_person_class
                )
            )
        if len(person_ids) > self.num_person_class:
            raise ValueError(
                'Training data contains more person IDs than the adversarial head supports'
            )
        self.person_id_to_class = {
            person_id: class_id for class_id, person_id in enumerate(person_ids)
        }
        mapped_ids = np.array(
            [self.person_id_to_class[person_id] for person_id in parsed_person_ids],
            dtype=np.int64
        )
        self.person_class_count = np.bincount(
            mapped_ids, minlength=self.num_person_class
        )
        present = self.person_class_count > 0
        class_weight = np.zeros(self.num_person_class, dtype=np.float32)
        class_weight[present] = (
            len(mapped_ids)
            / (self.num_person_class * self.person_class_count[present])
        )
        self.person_loss_func = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weight, device=self.device)
        )
        self.person_majority_rate = (
            self.person_class_count.max() / self.person_class_count.sum()
        )
        logging.info(
            'Person adversarial training: {} identities, GRL max-{:.3f}, '
            'gamma-{:.1f}, loss weight-{:.2f}, majority prior-{:.2%}'.format(
                len(person_ids), self.grl_max, self.grl_gamma, self.person_loss_weight,
                self.person_majority_rate
            )
        )

    def _person_targets(self, names):
        targets = []
        for name in names:
            person_id = self._parse_ntu_person_id(name)
            if person_id not in self.person_id_to_class:
                raise ValueError(
                    'Person ID {} is not in the training identity mapping'.format(person_id)
                )
            targets.append(self.person_id_to_class[person_id])
        return torch.tensor(targets, dtype=torch.long, device=self.device)

    def _grl_lambda(self):
        total_steps = max(
            1, self.max_epoch * self.optimizer_steps_per_epoch - 1
        )
        progress = min(float(self.global_step) / total_steps, 1.0)
        schedule = 2.0 / (1.0 + np.exp(-self.grl_gamma * progress)) - 1.0
        return self.grl_max * schedule

    def _init_geometry_consistency(self):
        if hasattr(self, 'geometry_consistency'):
            return

        training_args = getattr(self.args, 'training_args', {}) or {}
        self.geometry_consistency = bool(
            training_args.get('geometry_consistency', False)
        )
        paired_feeder = bool(
            getattr(self.feeders['train'], 'counterfactual_pair', False)
        )
        if paired_feeder != self.geometry_consistency:
            raise ValueError(
                'counterfactual_pair and geometry_consistency must be enabled together'
            )
        if not self.geometry_consistency:
            return
        if self.person_adversarial:
            raise ValueError(
                'Paired geometry consistency cannot be combined with person adversarial training'
            )

        self.consistency_js_weight = float(
            training_args.get('consistency_js_weight', 0.5)
        )
        self.consistency_feature_weight = float(
            training_args.get('consistency_feature_weight', 0.1)
        )
        self.consistency_temperature = float(
            training_args.get('consistency_temperature', 1.0)
        )
        self.consistency_warmup_epochs = int(
            training_args.get('consistency_warmup_epochs', 10)
        )
        self.consistency_ramp_epochs = int(
            training_args.get('consistency_ramp_epochs', 10)
        )
        self.consistency_primary_slot = int(
            training_args.get('consistency_primary_slot', 0)
        )
        if self.consistency_js_weight < 0.0:
            raise ValueError('consistency_js_weight must be non-negative')
        if self.consistency_feature_weight < 0.0:
            raise ValueError('consistency_feature_weight must be non-negative')
        if self.consistency_temperature <= 0.0:
            raise ValueError('consistency_temperature must be positive')
        if self.consistency_warmup_epochs < 0:
            raise ValueError('consistency_warmup_epochs must be non-negative')
        if self.consistency_ramp_epochs < 0:
            raise ValueError('consistency_ramp_epochs must be non-negative')
        if not 0 <= self.consistency_primary_slot < self.data_shape[-1]:
            raise ValueError('consistency_primary_slot is outside the person axis')

        logging.info(
            'Counterfactual geometry consistency: JS-{:.3f}, feature-{:.3f}, '
            'temperature-{:.2f}, warmup-{} epochs, ramp-{} epochs, person-slot-{}'.format(
                self.consistency_js_weight,
                self.consistency_feature_weight,
                self.consistency_temperature,
                self.consistency_warmup_epochs,
                self.consistency_ramp_epochs,
                self.consistency_primary_slot,
            )
        )

    def _geometry_consistency_scale(self):
        return linear_consistency_scale(
            self.global_step,
            self.optimizer_steps_per_epoch,
            self.consistency_warmup_epochs,
            self.consistency_ramp_epochs,
        )

    def train(self, epoch):
        self._init_person_adversarial()
        self._init_geometry_consistency()
        if hasattr(self.feeders['train'], 'set_epoch'):
            self.feeders['train'].set_epoch(epoch)
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        counterfactual_top1 = 0
        person_top1, person_sample = 0, 0
        person_class_correct = np.zeros(self.num_person_class, dtype=np.int64) \
            if self.person_adversarial else None
        person_class_total = np.zeros(self.num_person_class, dtype=np.int64) \
            if self.person_adversarial else None
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        self.optimizer.zero_grad()
        for num, (x, y, names) in enumerate(train_iter):
            if num >= self.train_micro_batches_per_epoch:
                break

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)

            # Calculating Output
            if self.geometry_consistency:
                if x.ndim != 7 or x.size(1) != 2:
                    raise ValueError(
                        'Paired geometry consistency expects input [N, 2, I, C, T, V, M], got {}'.format(
                            tuple(x.shape)
                        )
                    )
                batch_size = x.size(0)
                # Flatten the [clean, counterfactual] axis per sample so every
                # DataParallel shard receives both domains instead of one
                # shard seeing only clean and another only counterfactual data.
                paired_input = x.reshape(
                    batch_size * 2, *x.shape[2:]
                )
                (
                    paired_out,
                    _,
                    paired_fusion_feature,
                    paired_consistency_out,
                ) = self.model(paired_input, return_consistency=True)
                paired_out = paired_out.reshape(batch_size, 2, -1)
                out = paired_out[:, 0]
                counterfactual_out = paired_out[:, 1]
                paired_consistency_out = paired_consistency_out.reshape(
                    batch_size, 2, -1
                )
                clean_consistency_out = paired_consistency_out[:, 0]
                counterfactual_consistency_out = paired_consistency_out[:, 1]
                paired_fusion_feature = paired_fusion_feature.reshape(
                    batch_size,
                    2,
                    *paired_fusion_feature.shape[1:]
                )
                clean_fusion = paired_fusion_feature[:, 0]
                counterfactual_fusion = paired_fusion_feature[:, 1]
                clean_embedding = clean_fusion[
                    ..., self.consistency_primary_slot
                ]
                counterfactual_embedding = counterfactual_fusion[
                    ..., self.consistency_primary_slot
                ]

                clean_action_loss = self.loss_func(out, y)
                counterfactual_action_loss = self.loss_func(
                    counterfactual_out, y
                )
                action_loss = 0.5 * (
                    clean_action_loss + counterfactual_action_loss
                )
                js_loss = symmetric_js_divergence(
                    clean_consistency_out,
                    counterfactual_consistency_out,
                    self.consistency_temperature,
                )
                feature_loss = symmetric_stop_gradient_cosine(
                    clean_embedding, counterfactual_embedding
                )
                consistency_scale = self._geometry_consistency_scale()
                loss = action_loss + consistency_scale * (
                    self.consistency_js_weight * js_loss
                    + self.consistency_feature_weight * feature_loss
                )
            elif self.person_adversarial:
                grl_lambda = self._grl_lambda()
                out, _, person_out = self.model(
                    x, grl_lambda=grl_lambda, return_person=True
                )
                person_y = self._person_targets(names)
                action_loss = self.loss_func(out, y)
                person_loss = self.person_loss_func(person_out, person_y)
                loss = action_loss + self.person_loss_weight * person_loss
            else:
                out, _ = self.model(x)
                action_loss = loss = self.loss_func(out, y)

            # Updating Weights
            (loss / self.gradient_accumulation_steps).backward()
            optimizer_step = (
                (num + 1) % self.gradient_accumulation_steps == 0
            )
            if optimizer_step:
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1
                self.optimizer.zero_grad()

            # Calculating Recognition Accuracies
            num_sample += y.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()
            if self.geometry_consistency:
                counterfactual_reco_top1 = counterfactual_out.max(1)[1]
                counterfactual_top1 += counterfactual_reco_top1.eq(y).sum().item()
            elif self.person_adversarial:
                person_sample += y.size(0)
                person_reco_top1 = person_out.max(1)[1]
                person_top1 += person_reco_top1.eq(person_y).sum().item()
                person_y_cpu = person_y.detach().cpu().numpy()
                person_reco_cpu = person_reco_top1.detach().cpu().numpy()
                person_class_total += np.bincount(
                    person_y_cpu, minlength=self.num_person_class
                )
                person_class_correct += np.bincount(
                    person_y_cpu[person_reco_cpu == person_y_cpu],
                    minlength=self.num_person_class
                )

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                micro_step = (
                    epoch * self.train_micro_batches_per_epoch + num + 1
                )
                self.scalar_writer.add_scalar('learning_rate', lr, micro_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), micro_step)
                if self.geometry_consistency:
                    self.scalar_writer.add_scalar(
                        'train_action_loss', action_loss.item(), micro_step
                    )
                    self.scalar_writer.add_scalar(
                        'train_clean_action_loss',
                        clean_action_loss.item(), micro_step
                    )
                    self.scalar_writer.add_scalar(
                        'train_counterfactual_action_loss',
                        counterfactual_action_loss.item(), micro_step
                    )
                    self.scalar_writer.add_scalar(
                        'train_consistency_js', js_loss.item(), micro_step
                    )
                    self.scalar_writer.add_scalar(
                        'train_consistency_feature',
                        feature_loss.item(), micro_step
                    )
                    self.scalar_writer.add_scalar(
                        'train_consistency_scale',
                        consistency_scale, micro_step
                    )
                elif self.person_adversarial:
                    self.scalar_writer.add_scalar(
                        'train_action_loss', action_loss.item(), micro_step
                    )
                    self.scalar_writer.add_scalar(
                        'train_person_loss', person_loss.item(), micro_step
                    )
                    self.scalar_writer.add_scalar(
                        'grl_lambda', grl_lambda, micro_step
                    )
            if self.no_progress_bar:
                if self.geometry_consistency:
                    logging.info(
                        'Epoch: {}/{}, Batch: {}/{}, Clean CE: {:.4f}, '
                        'CF CE: {:.4f}, JS: {:.5f}, Cos: {:.5f}, '
                        'Scale: {:.3f}, LR: {:.4f}'.format(
                            epoch+1, self.max_epoch, num+1,
                            self.train_micro_batches_per_epoch,
                            clean_action_loss.item(),
                            counterfactual_action_loss.item(), js_loss.item(),
                            feature_loss.item(), consistency_scale, lr
                        )
                    )
                elif self.person_adversarial:
                    logging.info(
                        'Epoch: {}/{}, Batch: {}/{}, Action loss: {:.4f}, '
                        'Person loss: {:.4f}, GRL: {:.4f}, LR: {:.4f}'.format(
                            epoch+1, self.max_epoch, num+1,
                            self.train_micro_batches_per_epoch,
                            action_loss.item(), person_loss.item(), grl_lambda, lr
                        )
                    )
                else:
                    logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                        epoch+1, self.max_epoch, num+1,
                        self.train_micro_batches_per_epoch, loss.item(), lr
                    ))
            else:
                if self.geometry_consistency:
                    train_iter.set_description(
                        'CE: {:.4f}, JS: {:.4f}, Cos: {:.4f}, S: {:.2f}, LR: {:.4f}'.format(
                            action_loss.item(), js_loss.item(),
                            feature_loss.item(), consistency_scale, lr
                        )
                    )
                elif self.person_adversarial:
                    train_iter.set_description(
                        'Action: {:.4f}, Person: {:.4f}, GRL: {:.3f}, LR: {:.4f}'.format(
                            action_loss.item(), person_loss.item(), grl_lambda, lr
                        )
                    )
                else:
                    train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Showing Train Results
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, time()-start_train_time
        ))
        if self.geometry_consistency:
            counterfactual_acc = counterfactual_top1 / num_sample
            if self.scalar_writer:
                self.scalar_writer.add_scalar(
                    'train_counterfactual_acc',
                    counterfactual_acc, self.global_step
                )
            logging.info(
                'Epoch: {}/{}, Counterfactual training accuracy: '
                '{:d}/{:d}({:.2%})'.format(
                    epoch+1, self.max_epoch, counterfactual_top1,
                    num_sample, counterfactual_acc
                )
            )
        elif self.person_adversarial:
            person_acc = person_top1 / person_sample
            present = person_class_total > 0
            person_balanced_acc = np.mean(
                person_class_correct[present] / person_class_total[present]
            )
            if self.scalar_writer:
                self.scalar_writer.add_scalar(
                    'train_person_acc', person_acc, self.global_step
                )
                self.scalar_writer.add_scalar(
                    'train_person_balanced_acc', person_balanced_acc, self.global_step
                )
            logging.info(
                'Epoch: {}/{}, Person-head accuracy: {:d}/{:d}({:.2%}), '
                'majority prior: {:.2%}'.format(
                    epoch+1, self.max_epoch, person_top1, person_sample, person_acc,
                    self.person_majority_rate
                )
            )
            logging.info(
                'Epoch: {}/{}, Person-head balanced accuracy: {:.2%}, '
                'balanced chance: {:.2%}'.format(
                    epoch+1, self.max_epoch, person_balanced_acc,
                    1.0 / self.num_person_class
                )
            )
        logging.info('')

    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, _) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Calculating Output
                out, _ = self.model(x)

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out,5)[1]
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        return acc_top1, acc_top5, cm

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                U.validate_resume_checkpoint(
                    checkpoint,
                    self.model_name,
                    self.experiment_fingerprint,
                )
                if checkpoint.get('experiment_metadata') is not None:
                    self.experiment_metadata = checkpoint[
                        'experiment_metadata'
                    ]
                    self._write_experiment_protocol()
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = (
                    start_epoch * self.optimizer_steps_per_epoch
                )
                rng_restored = U.restore_rng_state(checkpoint)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Random state restored: {}'.format(rng_restored))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(),
                    self.optimizer.state_dict(),
                    self.scheduler.state_dict(),
                    epoch + 1,
                    best_state,
                    is_best,
                    self.args.work_dir,
                    self.save_dir,
                    self.model_name,
                    experiment_fingerprint=self.experiment_fingerprint,
                    experiment_metadata=self.experiment_metadata,
                )
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        x, y, names = iter(self.eval_loader).next()
        location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        self.model.eval()
        out, feature = self.model(x.float().to(self.device))

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.classifier.fc.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                data=data, label=label, name=names, out=out, cm=self.cm,
                feature=feature, weight=weight, location=location
            )
        logging.info('Finish extracting!')
        logging.info('')
