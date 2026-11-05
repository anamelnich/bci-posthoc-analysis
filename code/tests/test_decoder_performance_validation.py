"""Reject corrupt numeric decoder fields without truncation or index conflation."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from posthoc_analysis.decoder import _validate_decoder_struct

class DecoderPerformanceValidationTests(unittest.TestCase):
    def setUp(self):
        p=SimpleNamespace(labels=np.array([0.,1.,0.,1.]),posteriors=np.array([.2,.8,.3,.7]),
                          nTrials=2,file_id=np.array([1,2]),tpr=1.,tnr=1.,history=np.array([]))
        p._fieldnames=list(vars(p))
        self.decoder=SimpleNamespace(subjectID='e21',Classes=np.array([0.,1.]),performance=p)
        self.decoder._fieldnames=list(vars(self.decoder))
    def validate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return _validate_decoder_struct(self.decoder,'decoderR',Path('synthetic.mat'),'e21')
    def test_distinct_evaluation_and_retained_index_spaces(self):
        result=self.validate()
        self.assertEqual(result['n_labels'],4)
        self.assertEqual(result['n_trials_retained'],2)
    def test_fractional_or_nonfinite_labels_rejected(self):
        for value in [.5,np.nan,np.inf,-.5]:
            self.decoder.performance.labels[0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.validate()
    def test_fractional_classes_rejected(self):
        self.decoder.Classes=np.array([.4,1.])
        with self.assertRaisesRegex(ValueError,'Classes'):self.validate()
    def test_noninteger_or_nonscalar_retained_count_rejected(self):
        for value in [1.5,[2,2],np.nan]:
            self.decoder.performance.nTrials=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.validate()
    def test_retained_run_ids_rejected(self):
        for ids in [[1],[1,np.nan],[0,1],[1,1.5]]:
            self.decoder.performance.file_id=np.array(ids)
            with self.subTest(ids=ids),self.assertRaisesRegex(ValueError,'file_id'):self.validate()

if __name__=='__main__':unittest.main()
