import hashlib, logging, os, pickle, re, numpy as np
from torch.utils.data import Dataset


NTU_SAMPLE_PATTERN = re.compile(
    r'^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})'
    r'R(?P<repetition>\d{3})A(?P<action>\d{3})'
    r'(?:\.skeleton)?$',
    re.IGNORECASE,
)
NTU_SYMMETRIC_BONES = (
    (4, 8), (5, 9), (6, 10), (7, 11),
    (21, 23), (22, 24),
    (12, 16), (13, 17), (14, 18), (15, 19),
)


class NTU_Feeder(Dataset):
    def __init__(
        self, phase, dataset_path, inputs, num_frame, connect_joint, debug,
        bone_aug_prob=0.0, bone_aug_range=0.1,
        bone_normalization='none', canonical_bone_path='',
        canonical_bone_mix=1.0, canonical_bone_clip=(0.5, 2.0),
        eval_bone_style='none', eval_bone_style_strength=0.0,
        eval_bone_style_seed=20260721, eval_bone_style_id=0,
        counterfactual_pair=False, counterfactual_probability=1.0,
        counterfactual_seed=20260815, counterfactual_profile_samples=64,
        counterfactual_max_action=49,
        counterfactual_mix_range=(0.5, 1.0),
        counterfactual_ratio_clip=(0.5, 2.0),
        neighbor_mode='none', neighbor_calibration='', **kwargs
    ):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        self.phase = phase
        if neighbor_mode not in ('none', 'random', 'swap'):
            raise ValueError('neighbor_mode must be none/random/swap')
        self.neighbor_mode = neighbor_mode if phase == 'train' else 'none'
        if neighbor_mode != 'none' and (counterfactual_pair or bone_aug_prob > 0 or bone_normalization != 'none'):
            raise ValueError('Single-view neighbor control cannot stack paired/ordinary/canonical augmentation')
        self.neighbor_calibration_sha256 = ''
        if self.neighbor_mode == 'swap' and neighbor_calibration:
            import json
            with open(neighbor_calibration, 'rb') as handle:
                content = handle.read()
            calibration = json.loads(content)
            self.neighbor_calibration_sha256 = hashlib.sha256(content).hexdigest()
            counterfactual_mix_range = [calibration['mix']] * 2
            self._neighbor_expected_profile = calibration['profile_sha256']
        self.bone_aug_prob = bone_aug_prob if phase == 'train' else 0.0
        self.bone_aug_range = bone_aug_range
        self.bone_normalization = bone_normalization
        self.canonical_bone_path = canonical_bone_path
        self.canonical_bone_mix = float(canonical_bone_mix)
        self.canonical_bone_clip = tuple(float(value) for value in canonical_bone_clip)
        self.eval_bone_style = eval_bone_style if phase == 'eval' else 'none'
        self.eval_bone_style_strength = float(eval_bone_style_strength)
        self.eval_bone_style_seed = int(eval_bone_style_seed)
        self.eval_bone_style_id = int(eval_bone_style_id)
        self.counterfactual_pair = bool(
            counterfactual_pair and phase == 'train'
        )
        self.counterfactual_probability = float(counterfactual_probability)
        self.counterfactual_seed = int(counterfactual_seed)
        self.counterfactual_profile_samples = int(
            counterfactual_profile_samples
        )
        self.counterfactual_max_action = int(counterfactual_max_action)
        self.counterfactual_mix_range = tuple(
            float(value) for value in counterfactual_mix_range
        )
        self.counterfactual_ratio_clip = tuple(
            float(value) for value in counterfactual_ratio_clip
        )
        self.counterfactual_epoch = 0
        if not 0.0 <= self.bone_aug_prob <= 1.0:
            raise ValueError('bone_aug_prob must be in [0, 1]')
        if not 0.0 <= self.bone_aug_range < 1.0:
            raise ValueError('bone_aug_range must be in [0, 1)')
        if self.bone_normalization not in ('none', 'canonical'):
            raise ValueError("bone_normalization must be 'none' or 'canonical'")
        if not 0.0 <= self.canonical_bone_mix <= 1.0:
            raise ValueError('canonical_bone_mix must be in [0, 1]')
        if (
            len(self.canonical_bone_clip) != 2
            or self.canonical_bone_clip[0] <= 0.0
            or self.canonical_bone_clip[0] > self.canonical_bone_clip[1]
        ):
            raise ValueError('canonical_bone_clip must contain two ordered positive values')
        if self.eval_bone_style not in ('none', 'fixed'):
            raise ValueError("eval_bone_style must be 'none' or 'fixed'")
        if self.eval_bone_style_strength < 0.0:
            raise ValueError('eval_bone_style_strength must be non-negative')
        if self.counterfactual_seed < 0:
            raise ValueError('counterfactual_seed must be non-negative')
        if (
            not np.isfinite(self.counterfactual_probability)
            or not 0.0 <= self.counterfactual_probability <= 1.0
        ):
            raise ValueError('counterfactual_probability must be in [0, 1]')
        if self.counterfactual_profile_samples <= 0:
            raise ValueError('counterfactual_profile_samples must be positive')
        if not 1 <= self.counterfactual_max_action <= 49:
            raise ValueError('counterfactual_max_action must be in [1, 49]')
        if (
            len(self.counterfactual_mix_range) != 2
            or not np.all(np.isfinite(self.counterfactual_mix_range))
            or self.counterfactual_mix_range[0] < 0.0
            or self.counterfactual_mix_range[0]
            > self.counterfactual_mix_range[1]
        ):
            raise ValueError(
                'counterfactual_mix_range must contain two ordered non-negative values'
            )
        if (
            len(self.counterfactual_ratio_clip) != 2
            or not np.all(np.isfinite(self.counterfactual_ratio_clip))
            or self.counterfactual_ratio_clip[0] <= 0.0
            or self.counterfactual_ratio_clip[0]
            > self.counterfactual_ratio_clip[1]
        ):
            raise ValueError(
                'counterfactual_ratio_clip must contain two ordered positive values'
            )
        if self.counterfactual_pair and self.bone_aug_prob > 0.0:
            raise ValueError(
                'counterfactual_pair requires bone_aug_prob=0 so view 0 remains clean'
            )
        self.bone_traversal = self._build_bone_traversal()
        self.canonical_bone_lengths = self._load_canonical_bone_lengths()
        self.eval_bone_style_profile = self._build_eval_bone_style_profile()
        self.eval_bone_style_ratios = np.exp(
            self.eval_bone_style_strength * self.eval_bone_style_profile
        ).astype(np.float32)
        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        self.data_path = data_path
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        try:
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()

        self.counterfactual_profile_subject_ids = np.empty(0, dtype=np.int64)
        self.counterfactual_profiles = np.empty(
            (0, len(self.conn)), dtype=np.float32
        )
        self.counterfactual_profile_counts = np.empty(0, dtype=np.int64)
        self.counterfactual_profile_capture_subject_ids = np.empty(
            0, dtype=np.int64
        )
        self.counterfactual_profile_capture_names = np.empty(
            0, dtype='<U1'
        )
        self.counterfactual_profile_sha256 = ''
        if self.counterfactual_pair or self.neighbor_mode == 'swap':
            (
                self.counterfactual_profile_subject_ids,
                self.counterfactual_profiles,
                self.counterfactual_profile_counts,
                self.counterfactual_profile_capture_subject_ids,
                self.counterfactual_profile_capture_names,
            ) = self._build_counterfactual_profile_bank()
            self.counterfactual_profile_sha256 = self._profile_bank_sha256()
            if hasattr(self, '_neighbor_expected_profile') and self._neighbor_expected_profile != self.counterfactual_profile_sha256:
                raise ValueError('Neighbor calibration/profile hash mismatch')

        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.name = self.name[:300]
            self.seq_len = self.seq_len[:300]
        if self.bone_aug_prob > 0:
            logging.info(
                'Bone-length augmentation: probability-{}, range-{}'.format(
                    self.bone_aug_prob, self.bone_aug_range
                )
            )
        if self.bone_normalization == 'canonical':
            logging.info(
                'Canonical bone normalization: mix-{:.2f}, clip-[{:.2f}, {:.2f}], template-{}'.format(
                    self.canonical_bone_mix,
                    self.canonical_bone_clip[0],
                    self.canonical_bone_clip[1],
                    self.canonical_bone_path,
                )
            )
        if self.eval_bone_style == 'fixed' and self.eval_bone_style_strength > 0:
            logging.info(
                'Evaluation geometry style: id-{}, seed-{}, strength-{:.3f}, '
                'ratio range-[{:.3f}, {:.3f}]'.format(
                    self.eval_bone_style_id,
                    self.eval_bone_style_seed,
                    self.eval_bone_style_strength,
                    self.eval_bone_style_ratios.min(),
                    self.eval_bone_style_ratios.max(),
                )
            )
        if self.counterfactual_pair:
            logging.info(
                'Counterfactual geometry pairs: {} train-only donor profiles, '
                'probability-{:.2f}, mix-[{:.2f}, {:.2f}], clip-[{:.2f}, {:.2f}], '
                'seed-{}, profile-sha256-{}'.format(
                    len(self.counterfactual_profile_subject_ids),
                    self.counterfactual_probability,
                    self.counterfactual_mix_range[0],
                    self.counterfactual_mix_range[1],
                    self.counterfactual_ratio_clip[0],
                    self.counterfactual_ratio_clip[1],
                    self.counterfactual_seed,
                    self.counterfactual_profile_sha256,
                )
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = np.load(self.data_path, mmap_mode='r')

    @staticmethod
    def _parse_ntu_sample_name(name):
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        basename = str(name).replace('\\', '/').rsplit('/', 1)[-1]
        match = NTU_SAMPLE_PATTERN.fullmatch(basename)
        if match is None:
            raise ValueError('Cannot parse NTU sample name: {}'.format(name))
        return {
            key: int(match.group(key))
            for key in ('setup', 'camera', 'subject', 'repetition', 'action')
        }

    @staticmethod
    def _symmetrize_ntu_profile(profile):
        profile = np.asarray(profile, dtype=np.float32).copy()
        if profile.shape == (25,):
            for left, right in NTU_SYMMETRIC_BONES:
                if profile[left] > 0.0 and profile[right] > 0.0:
                    value = float(np.sqrt(profile[left] * profile[right]))
                    profile[left] = value
                    profile[right] = value
        return profile

    def _build_counterfactual_profile_bank(self):
        """Fit deterministic, train-only primary-person bone profiles."""
        captures = {}
        for index, (name, zero_based_label) in enumerate(
            zip(self.name, self.label)
        ):
            identifiers = self._parse_ntu_sample_name(name)
            if identifiers['action'] != int(zero_based_label) + 1:
                raise ValueError(
                    'Filename/label mismatch at index {}: A{:03d} versus {}'.format(
                        index, identifiers['action'], zero_based_label
                    )
                )
            if identifiers['action'] > self.counterfactual_max_action:
                continue

            subject = identifiers['subject']
            capture_key = (
                identifiers['setup'],
                subject,
                identifiers['repetition'],
                identifiers['action'],
            )
            subject_captures = captures.setdefault(subject, {})
            previous = subject_captures.get(capture_key)
            if previous is None or identifiers['camera'] < previous[0]:
                subject_captures[capture_key] = (
                    identifiers['camera'], index
                )

        roots = np.arange(len(self.conn)) == np.asarray(self.conn)
        subject_ids, profiles, counts = [], [], []
        capture_subject_ids, capture_names = [], []
        for subject, subject_captures in sorted(captures.items()):
            ranked = []
            for capture_key, (_, index) in subject_captures.items():
                token = '{}|{}|{}'.format(
                    self.counterfactual_seed, subject, capture_key
                ).encode('utf-8')
                ranked.append((hashlib.sha256(token).digest(), index))
            ranked.sort()
            selected_indices = [
                index
                for _, index in ranked[:self.counterfactual_profile_samples]
            ]
            for index in selected_indices:
                capture_subject_ids.append(subject)
                capture_name = self.name[index]
                if isinstance(capture_name, bytes):
                    capture_name = capture_name.decode('utf-8')
                capture_names.append(
                    str(capture_name).replace('\\', '/').rsplit('/', 1)[-1]
                )

            observations = []
            for index in selected_indices:
                data = np.array(self.data[index])[:, :self.T, :, :]
                data = self.normalize_bone_lengths(
                    data, self.seq_len[index]
                )
                lengths = self._sample_bone_lengths(
                    data, self.seq_len[index]
                )[:, 0]
                if np.sum(np.isfinite(lengths[~roots])) == np.sum(~roots):
                    observations.append(lengths)
            if not observations:
                raise RuntimeError(
                    'No valid counterfactual profile samples for subject {}'.format(
                        subject
                    )
                )

            stacked = np.stack(observations)
            profile = np.zeros(len(self.conn), dtype=np.float32)
            profile[~roots] = np.nanmedian(
                stacked[:, ~roots], axis=0
            ).astype(np.float32)
            profile = self._symmetrize_ntu_profile(profile)
            if (
                not np.all(np.isfinite(profile[~roots]))
                or np.any(profile[~roots] <= 0.0)
            ):
                raise RuntimeError(
                    'Invalid counterfactual profile for subject {}'.format(
                        subject
                    )
                )
            subject_ids.append(subject)
            profiles.append(profile)
            counts.append(len(observations))

        if len(subject_ids) < 2:
            raise RuntimeError(
                'Counterfactual pairing requires at least two training subjects'
            )
        return (
            np.asarray(subject_ids, dtype=np.int64),
            np.stack(profiles).astype(np.float32),
            np.asarray(counts, dtype=np.int64),
            np.asarray(capture_subject_ids, dtype=np.int64),
            np.asarray(capture_names),
        )

    def _profile_bank_sha256(self):
        digest = hashlib.sha256()
        for array in (
            np.asarray(self.conn, dtype=np.int64),
            self.counterfactual_profile_subject_ids,
            self.counterfactual_profiles,
            self.counterfactual_profile_counts,
            self.counterfactual_profile_capture_subject_ids,
            self.counterfactual_profile_capture_names,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode('ascii'))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def set_epoch(self, epoch):
        """Set the epoch used by stateless counterfactual sampling."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError('epoch must be non-negative')
        self.counterfactual_epoch = epoch

    def _counterfactual_rng(self, idx, name):
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        stable_name = str(name).replace('\\', '/').rsplit('/', 1)[-1]
        name_digest = hashlib.sha256(stable_name.encode('utf-8')).digest()
        name_words = np.frombuffer(name_digest[:8], dtype='<u4')
        seed_sequence = np.random.SeedSequence([
            self.counterfactual_seed,
            self.counterfactual_epoch,
            int(idx),
            int(name_words[0]),
            int(name_words[1]),
        ])
        return np.random.default_rng(seed_sequence)

    def counterfactual_bone_transfer(self, data, seq_len, idx, name):
        """Retarget only the primary actor to a real training-subject profile."""
        if not self.counterfactual_pair and self.neighbor_mode != 'swap':
            return data

        rng = self._counterfactual_rng(idx, name)
        if rng.random() >= self.counterfactual_probability:
            return data.copy()

        identifiers = self._parse_ntu_sample_name(name)
        # NTU preprocessing energy-sorts the two actor slots. For interaction
        # actions A050-A060, filename Pxxx therefore cannot identify slot 0
        # reliably. Keep those pairs identical and apply the controlled
        # geometry objective only to the single-person A001-A049 protocol.
        if identifiers['action'] > self.counterfactual_max_action:
            return data.copy()

        source_subject = identifiers['subject']
        eligible = np.flatnonzero(
            self.counterfactual_profile_subject_ids != source_subject
        )
        if eligible.size == 0:
            raise RuntimeError(
                'No counterfactual donor remains after excluding source subject {}'
                .format(source_subject)
            )
        donor_index = int(eligible[rng.integers(0, eligible.size)])
        target = self.counterfactual_profiles[donor_index]

        current = self._sample_bone_lengths(data, seq_len)[:, 0]
        roots = np.arange(len(self.conn)) == np.asarray(self.conn)
        valid = (
            ~roots
            & np.isfinite(current)
            & (current > 1e-6)
            & np.isfinite(target)
            & (target > 0.0)
        )
        bone_scale = np.ones((len(self.conn), 1), dtype=np.float32)
        ratio = np.ones(len(self.conn), dtype=np.float32)
        np.divide(target, current, out=ratio, where=valid)
        ratio = np.clip(
            ratio,
            self.counterfactual_ratio_clip[0],
            self.counterfactual_ratio_clip[1],
        )
        if self.counterfactual_mix_range[0] == self.counterfactual_mix_range[1]:
            mix = self.counterfactual_mix_range[0]
        else:
            mix = float(rng.uniform(
                self.counterfactual_mix_range[0],
                self.counterfactual_mix_range[1],
            ))
        bone_scale[valid, 0] = np.power(ratio[valid], mix)

        output = data.copy()
        primary = data[:, :, :, :1]
        styled_primary = self._rescale_bones(primary, bone_scale)
        output[:, :, :, :1] = styled_primary
        return output

    def neighbor_augment(self, data, seq_len, idx, name):
        """Single view, A001-A049 primary slot only; stateless per epoch."""
        if self.neighbor_mode == 'none':
            return data
        if self.neighbor_mode == 'swap':
            return self.counterfactual_bone_transfer(data, seq_len, idx, name)
        rng = self._counterfactual_rng(idx, name)
        if rng.random() >= self.counterfactual_probability:
            return data.copy()
        if self._parse_ntu_sample_name(name)['action'] > self.counterfactual_max_action:
            return data.copy()
        scale = rng.uniform(1-self.bone_aug_range/2, 1+self.bone_aug_range/2)
        scales = scale * rng.uniform(1-self.bone_aug_range, 1+self.bone_aug_range, (len(self.conn), 1))
        output = data.copy()
        output[:, :, :, :1] = self._rescale_bones(data[:, :, :, :1], scales)
        return output

    def _build_bone_traversal(self):
        """Return parent-before-child joint order for kinematic reconstruction."""
        num_joint = len(self.conn)
        children = [[] for _ in range(num_joint)]
        roots = []
        for joint, parent in enumerate(self.conn):
            if joint == parent:
                roots.append(joint)
            else:
                children[parent].append(joint)

        traversal, queue = [], list(roots)
        visited = set(roots)
        while queue:
            parent = queue.pop(0)
            for child in children[parent]:
                if child not in visited:
                    traversal.append(child)
                    visited.add(child)
                    queue.append(child)

        # Keep the feeder usable for a new dataset even if its parent array is
        # not a perfectly rooted tree.
        traversal.extend(joint for joint in range(num_joint) if joint not in visited)
        return traversal

    def _load_canonical_bone_lengths(self):
        if self.bone_normalization == 'none':
            return None
        if not self.canonical_bone_path or not os.path.isfile(self.canonical_bone_path):
            raise ValueError(
                'canonical_bone_path must point to a training-set bone template: {}'.format(
                    self.canonical_bone_path
                )
            )

        template_file = np.load(self.canonical_bone_path, allow_pickle=False)
        try:
            if isinstance(template_file, np.lib.npyio.NpzFile):
                if 'bone_lengths' not in template_file.files:
                    raise ValueError('Canonical template does not contain bone_lengths')
                lengths = np.array(template_file['bone_lengths'], dtype=np.float32)
                if 'connect_joint' in template_file.files:
                    saved_conn = np.array(template_file['connect_joint'], dtype=np.int64)
                    if not np.array_equal(saved_conn, self.conn):
                        raise ValueError('Canonical template skeleton topology does not match the feeder')
            else:
                lengths = np.array(template_file, dtype=np.float32)
        finally:
            if isinstance(template_file, np.lib.npyio.NpzFile):
                template_file.close()

        if lengths.shape != (len(self.conn),):
            raise ValueError(
                'Canonical bone template must have shape ({},), got {}'.format(
                    len(self.conn), lengths.shape
                )
            )
        non_root = np.arange(len(self.conn)) != self.conn
        if not np.all(np.isfinite(lengths[non_root])) or np.any(lengths[non_root] <= 0.0):
            raise ValueError('Canonical template contains invalid non-root bone lengths')
        return lengths

    def _sample_bone_lengths(self, data, seq_len):
        """Estimate one robust length per bone and person from valid frames."""
        _, num_frame, num_joint, num_person = data.shape
        valid_frame_count = min(max(int(seq_len), 0), num_frame)
        output = np.full((num_joint, num_person), np.nan, dtype=np.float32)
        if valid_frame_count == 0:
            return output

        sample = data[:, :valid_frame_count, :, :]
        parent = sample[:, :, self.conn, :]
        bone_length = np.linalg.norm(sample - parent, axis=0)
        person_valid = np.sum(np.abs(sample), axis=(0, 2)) > 1e-6
        valid = person_valid[:, None, :] & (bone_length > 1e-6)

        for joint in range(num_joint):
            if joint == self.conn[joint]:
                continue
            for person in range(num_person):
                values = bone_length[:, joint, person][valid[:, joint, person]]
                if values.size > 0:
                    output[joint, person] = np.median(values)
        return output

    def normalize_bone_lengths(self, data, seq_len):
        """Map every valid skeleton to the training-set canonical bone lengths."""
        if self.bone_normalization == 'none' or self.canonical_bone_mix == 0.0:
            return data

        current = self._sample_bone_lengths(data, seq_len)
        target = self.canonical_bone_lengths.reshape(-1, 1)
        valid = np.isfinite(current) & (current > 1e-6)
        bone_scale = np.ones_like(current, dtype=np.float32)
        ratio = np.ones_like(current, dtype=np.float32)
        np.divide(target, current, out=ratio, where=valid)
        ratio = np.clip(ratio, self.canonical_bone_clip[0], self.canonical_bone_clip[1])
        bone_scale[valid] = np.power(ratio[valid], self.canonical_bone_mix)
        return self._rescale_bones(data, bone_scale)

    def _build_eval_bone_style_profile(self):
        """Create one deterministic, anatomically symmetric geometry direction."""
        num_joint = len(self.conn)
        profile = np.zeros(num_joint, dtype=np.float32)
        if self.eval_bone_style != 'fixed':
            return profile

        # SeedSequence avoids dependence on Python's randomized hash and makes
        # the same style reproducible with any number of DataLoader workers.
        rng = np.random.default_rng(np.random.SeedSequence([
            self.eval_bone_style_seed, self.eval_bone_style_id
        ]))
        local_profile = rng.uniform(-1.0, 1.0, size=num_joint)

        # NTU RGB+D left/right corresponding bones. A shared value prevents
        # implausible asymmetry from dominating the robustness measurement.
        if num_joint == 25:
            symmetric_pairs = (
                (4, 8), (5, 9), (6, 10), (7, 11),
                (21, 23), (22, 24),
                (12, 16), (13, 17), (14, 18), (15, 19),
            )
            for left, right in symmetric_pairs:
                value = rng.uniform(-1.0, 1.0)
                local_profile[left] = value
                local_profile[right] = value

        # Mix overall stature with relative limb proportions. Normalization
        # gives strength a stable meaning: the largest log-length change is
        # exactly +/- strength for every style ID.
        global_profile = rng.uniform(-1.0, 1.0)
        profile = 0.7 * local_profile + 0.3 * global_profile
        roots = np.arange(num_joint) == self.conn
        profile[roots] = 0.0
        max_abs = np.max(np.abs(profile[~roots])) if np.any(~roots) else 0.0
        if max_abs > 0:
            profile /= max_abs
        return profile.astype(np.float32)

    def _rescale_bones(self, data, bone_scale):
        """Reconstruct joints parent-first using time-constant bone scales."""
        _, _, num_joint, num_person = data.shape
        scale = np.asarray(bone_scale, dtype=data.dtype)
        if scale.ndim == 1:
            scale = scale.reshape(num_joint, 1)
        if scale.shape not in ((num_joint, 1), (num_joint, num_person)):
            raise ValueError('bone_scale has an incompatible shape')

        output = data.copy()
        person_valid = np.sum(np.abs(data), axis=(0, 2)) > 1e-6
        for joint in self.bone_traversal:
            parent = self.conn[joint]
            if joint == parent:
                continue

            bone = data[:, :, joint, :] - data[:, :, parent, :]
            bone_valid = np.linalg.norm(bone, axis=0) > 1e-6
            valid = person_valid & bone_valid
            joint_scale = np.broadcast_to(scale[joint], (num_person,))
            candidate = (
                output[:, :, parent, :]
                + bone * joint_scale.reshape(1, 1, num_person)
            )
            output[:, :, joint, :] = np.where(
                valid.reshape(1, valid.shape[0], valid.shape[1]),
                candidate,
                data[:, :, joint, :],
            )
        return output

    def bone_length_augment(self, data):
        """Change static body geometry while preserving bone directions over time."""
        if self.bone_aug_prob == 0 or np.random.random() >= self.bone_aug_prob:
            return data

        _, _, num_joint, num_person = data.shape

        # Combine a per-person scale with per-bone scales, fixed across frames.
        # Positive scales preserve valid bone directions and frame order,
        # but change joint coordinates and may change inter-joint contacts.
        half_range = self.bone_aug_range * 0.5
        person_scale = np.random.uniform(
            1.0 - half_range, 1.0 + half_range, size=num_person
        )
        bone_scale = np.random.uniform(
            1.0 - self.bone_aug_range,
            1.0 + self.bone_aug_range,
            size=(num_joint, num_person),
        )
        bone_scale *= person_scale.reshape(1, num_person)

        return self._rescale_bones(data, bone_scale)

    def apply_eval_bone_style(self, data):
        """Apply a fixed counterfactual body style during evaluation only."""
        if self.eval_bone_style != 'fixed' or self.eval_bone_style_strength == 0:
            return data
        return self._rescale_bones(data, self.eval_bone_style_ratios)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]
        seq_len = self.seq_len[idx]

        data = data[:,:self.T,:,:]
        data = self.normalize_bone_lengths(data, seq_len)

        if self.counterfactual_pair:
            counterfactual = self.counterfactual_bone_transfer(
                data, seq_len, idx, name
            )
            clean_input = self._stack_inputs(data)
            counterfactual_input = self._stack_inputs(counterfactual)
            paired_input = np.stack(
                (clean_input, counterfactual_input), axis=0
            )
            return paired_input, label, name

        data = self.neighbor_augment(data, seq_len, idx, name)
        data = self.bone_length_augment(data)
        data = self.apply_eval_bone_style(data)

        return self._stack_inputs(data), label, name

    def _stack_inputs(self, data):
        """Build the configured J/V/B streams for one raw skeleton view."""

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone = self.multi_input(data)
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)
        return data_new

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
            velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
        for i in range(len(self.conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        return joint, velocity, bone


class NTU_Location_Feeder():
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 2, self.T, self.V, self.M))
        for i, name in enumerate(names):
            with open(name, 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        fr.readline()
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')
                            if joint < self.V and person < self.M:
                                location[i,0,frame,joint,person] = float(v[5])
                                location[i,1,frame,joint,person] = float(v[6])
        return location
