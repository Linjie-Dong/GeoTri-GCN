"""CPU-only synthetic smoke check. No training, dataset access or accuracy claim."""
from pathlib import Path
import sys
import copy
import tempfile
import pickle
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import yaml
from src import model
from src.dataset.graphs import Graph
from src.dataset.ntu_feeder import NTU_Feeder
from src.model.fusions import TriAxisStreamFusion, SelectiveMultiScaleTemporal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trusted-checkpoint', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1)
    np.random.seed(1)
    graph = Graph('ntu-xsub')
    checks = []
    for name in ('EfficientGCN-B0_NTU60', 'GeoTri-GCN_NTU60', 'EfficientGCN-B0_NTU120', 'GeoTri-GCN_NTU120'):
        config = yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text())
        classes = 120 if '120' in config['dataset'] else 60
        net = model.create(config['model_type'], **copy.deepcopy(config['model_args']),
                           data_shape=[3, 6, 288, 25, 2], num_class=classes,
                           A=torch.tensor(graph.A, dtype=torch.float32), parts=graph.parts).eval()
        before = {k:v.clone() for k,v in net.state_dict().items()}
        with torch.inference_mode():
            output, _ = net(torch.randn(1, 3, 6, 288, 25, 2))
        assert output.shape == (1, classes) and torch.isfinite(output).all()
        assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
        assert net.person_head is None
        if name.startswith('GeoTri-GCN'):
            assert net.stream_fusion is not None and net.temporal_refine is not None
        else:
            assert net.stream_fusion is None and net.temporal_refine is None
        record = {'config':name, 'output_shape':list(output.shape), 'state_unchanged':True}
        if name == 'GeoTri-GCN_NTU60' and args.trusted_checkpoint:
            checkpoint = torch.load(args.trusted_checkpoint, map_location='cpu', weights_only=False)
            net.load_state_dict(checkpoint['model'], strict=True)
            with torch.inference_mode():
                out,_ = net(torch.randn(1,3,6,288,25,2))
            assert torch.isfinite(out).all()
            record['trusted_checkpoint_strict_load'] = True
        checks.append(record)

    fusion = TriAxisStreamFusion(3, torch.tensor(graph.A)).eval()
    xs = [torch.randn(2,32,16,25) for _ in range(3)]
    assert torch.equal(fusion(xs), torch.cat(xs,1))
    refine = SelectiveMultiScaleTemporal(96).eval()
    x = torch.cat(xs,1)
    assert torch.equal(refine(x), x)

    # Exercise the data path with a temporary synthetic sequence, never real NTU data.
    with tempfile.TemporaryDirectory(prefix='geotri_smoke_') as tmp:
        folder = Path(tmp)
        raw = np.random.normal(size=(2,3,300,25,2)).astype(np.float32)
        for phase in ('train','eval'):
            np.save(folder/f'{phase}_data.npy',raw)
            with (folder/f'{phase}_label.pkl').open('wb') as f:
                pickle.dump((['S001C001P001R001A001','S001C001P002R001A002'],[0,1],[300,300]),f)
            feeder = NTU_Feeder(phase,str(folder),'JVB',288,graph.connect_joint,False,
                                bone_aug_prob=1.0,bone_aug_range=0.1)
            values, label, name = feeder[0]
            assert values.shape == (3,6,288,25,2) and np.isfinite(values).all()
            assert feeder.bone_aug_prob == (1.0 if phase == 'train' else 0.0)
            assert not feeder.counterfactual_pair and feeder.neighbor_mode == 'none'
            feeder.data._mmap.close()
            del feeder
    report = {'status':'passed','device':'cpu','training_executed':False,
              'real_dataset_accessed':False,'torch':torch.__version__,
              'models':checks,'zero_strength_identity':True,'synthetic_feeder':True}
    if args.report:
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
