import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paired_statistics import paired_bootstrap
class PairedStatisticsTests(unittest.TestCase):
    def test_pairing_and_document_contract(self):
        a=[{'question_id':'1','document_id':'a','answers':['yes'],'prediction':'no'},
           {'question_id':'2','document_id':'b','answers':['yes'],'prediction':'no'}]
        b=[dict(row,prediction='yes') for row in reversed(a)]
        result=paired_bootstrap(a,b,samples=100)
        self.assertEqual(result['metrics']['exact_match'],{'delta':1.0,'ci95':[1.0,1.0]})
        self.assertEqual(result,paired_bootstrap(a,b,samples=100))
        with self.assertRaises(ValueError): paired_bootstrap(a,b[:1])
        a[1]['document_id']='a'
        with self.assertRaises(ValueError): paired_bootstrap(a,b)
if __name__=='__main__': unittest.main()
