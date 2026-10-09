import base64,io,sys,unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from demo_input import parse_request
class DemoInputTests(unittest.TestCase):
    def test_image_and_invalid_input(self):
        output=io.BytesIO();Image.new('RGB',(20,30)).save(output,format='PNG')
        payload={'image_base64':base64.b64encode(output.getvalue()).decode(),'question':' date? ','max_image_edge':768}
        image,question,edge=parse_request(payload)
        self.assertEqual((image.size,question,edge),((20,30),'date?',768))
        for change in [{'question':''},{'image_base64':'%%%invalid'},{'max_image_edge':True},{'max_image_edge':1024}]:
            with self.assertRaises(ValueError):parse_request({**payload,**change})
if __name__=='__main__':unittest.main()
