import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from train_all import main, build_policy, batch_distribution
from analyze import main as analyze_main


class ModelSelectionTests(unittest.TestCase):
    def test_classic_rollout_values_are_per_timestep(self):
        model=build_policy("classic",13,2)
        x=torch.randn(4,2,13)
        dist,value=batch_distribution(model,x,torch.ones(4,2,3,dtype=torch.bool))
        self.assertEqual(value.shape,(4,))
        for i in range(4):
            expected=model(x[i],torch.ones(2,3,dtype=torch.bool))[1]
            torch.testing.assert_close(value[i],expected)
        self.assertEqual(dist.probs.shape,(4,2,3))

    def test_train_and_analyze_both_cli_modes(self):
        times=pd.to_datetime([f"2026-{month:02d}-02 {clock}" for month in [1,6,9] for clock in ["10:00","10:01","10:02","15:55"]])
        prices=np.full((12,2,2),100.)
        features=np.zeros((12,2,9),dtype=np.float32)
        available=np.ones((12,2),dtype=bool)
        names=["VWAP","RSI","BB","Donchian","GP"]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for stock in ['A','B']: (root/f'{stock}.parquet').touch()
            report=root/'monthly.csv'
            pd.DataFrame({'stock':['A','B']}).to_csv(report,index=False)
            for choice in ['classic','joint','classic_lstm']:
                out=root/choice
                args=['train_all.py','--model','classic' if choice=='classic_lstm' else choice,'--data-dir',str(root),'--results',str(report),'--output',str(out),'--epochs','1','--device','cpu','--rollout','2']
                if choice=='classic_lstm': args += ['--use-lstm','--lookback','3']
                with patch.object(sys,'argv',args),patch('train_all.prepare',return_value=(times,prices,features,names,available)),contextlib.redirect_stdout(io.StringIO()):
                    main()
                saved=torch.load(out/'policy.pt',weights_only=True)
                self.assertEqual(saved['actions'],11 if choice=='joint' else 3)
                self.assertEqual(saved['inputs'],15 if choice=='classic_lstm' else (13 if choice=='classic' else 16))
                self.assertEqual(saved['model'],'classic' if choice=='classic_lstm' else choice)
                with patch.object(sys,'argv',['analyze.py','--run',str(out),'--max-explanations','2']),contextlib.redirect_stdout(io.StringIO()):
                    analyze_main()
                summary=json.loads((out/'analysis_test/summary.json').read_text())
                self.assertTrue(summary['replay_matches_original'])

if __name__=='__main__': unittest.main()
