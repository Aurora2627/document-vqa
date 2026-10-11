import sys,unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from region_reading import region_boxes,rank_regions,locator_messages,prepare_region_views
class RegionReadingTests(unittest.TestCase):
    def test_boxes_ranking_and_reference_free_contract(self):
        boxes=region_boxes((101,203))
        self.assertEqual(len(boxes),9)
        for left,top,right,bottom in boxes:
            self.assertTrue(0<=left<right<=101 and 0<=top<bottom<=203)
        ranking,probs=rank_regions([0,1,2,3,4,5,6,7,8]);self.assertEqual(ranking[:2],[9,8]);self.assertAlmostEqual(sum(probs),1,places=6)
        self.assertEqual(rank_regions([0]*9)[0],list(range(1,10)))
        with self.assertRaises(ValueError):rank_regions([float('nan')]*9)
        views,selected=prepare_region_views(Image.new('RGB',(101,203)),[9,8],64)
        self.assertEqual((len(views),selected),(3,[list(boxes[8]),list(boxes[7])]))
        self.assertTrue(all(max(v.size)<=64 for v in views))
        with self.assertRaises(ValueError):prepare_region_views(Image.new('RGB',(20,20)),[1,1])
        self.assertIn('Where is the date?',locator_messages('Where is the date?')[0]['content'][1]['text'])
if __name__=='__main__':unittest.main()
