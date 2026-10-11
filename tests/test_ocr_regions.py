import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ocr_regions import retrieve_regions
class OCRRegionTests(unittest.TestCase):
    def test_question_retrieval_geometry_and_fallback(self):
        def item(text,y):return {'text':text,'box':[[5,y],[95,y],[95,y+5],[5,y+5]],'confidence':.9}
        rows=[item('INVOICE DATE',10),item('2026-10-11',20),item('TOTAL AMOUNT',80),item('42.00',90)]
        date=retrieve_regions(rows,'What is the invoice date?',(100,100),1)
        amount=retrieve_regions(rows,'What is the total amount?',(100,100),1)
        self.assertLess(date['regions'][0]['center_y'],amount['regions'][0]['center_y'])
        self.assertTrue(all(0<=region['box'][1]<region['box'][3]<=100 for region in date['regions']))
        self.assertEqual(retrieve_regions(rows,'unseenword',(100,100))['regions'],[])
        self.assertEqual(retrieve_regions([],'date',(100,100))['regions'],[])
if __name__=='__main__':unittest.main()
