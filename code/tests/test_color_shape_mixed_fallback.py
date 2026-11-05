"""Color/Shape fallback must retain the group-adjusted within-subject error."""
import contextlib,io,sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
from statsmodels.formula.api import ols
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from posthoc_analysis.color_shape import _mixed_anova_fallback_two_by_two
class ColorShapeFallbackTests(unittest.TestCase):
 def test_unequal_groups_against_independent_regression(self):
  rng=np.random.default_rng(121);rows=[]
  for group,n in [('bci',13),('control',12)]:
   for i in range(n):
    baseline=rng.normal(700,60);change=rng.normal(-60 if group=='bci' else 4,20)
    for session,val in [('pre',baseline),('post',baseline+change)]:rows.append(dict(subject_id=f'{group}{i}',group=group,session=session,rt=val))
  data=pd.DataFrame(rows)
  with contextlib.redirect_stdout(io.StringIO()):table=_mixed_anova_fallback_two_by_two(data,'rt','congruent').set_index('Source')
  wide=data.pivot(index=['subject_id','group'],columns='session',values='rt');change=(wide.post-wide.pre).rename('change').reset_index()
  model=ols('change ~ C(group)',change).fit()
  # Mean design row represents the participant-weighted marginal change.
  test=model.t_test(model.model.exog.mean(axis=0))
  self.assertAlmostEqual(table.loc['Session','F'],float(test.tvalue.item())**2,places=9)
  self.assertAlmostEqual(table.loc['Session','p-unc'],float(test.pvalue),places=10)
  self.assertEqual(table.loc['Session','DF2'],23)
  self.assertAlmostEqual(table.loc['Interaction','F'],float(model.tvalues.iloc[1])**2,places=9)
if __name__=='__main__':unittest.main()
