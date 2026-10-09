"""Validate local browser input before allocating model tensors."""
import base64, binascii, io
from PIL import Image

def parse_request(payload):
    question=payload.get('question')
    if not isinstance(question,str) or not question.strip() or len(question)>2000:
        raise ValueError('请输入 1–2000 字的问题')
    encoded=payload.get('image_base64')
    if not isinstance(encoded,str): raise ValueError('请上传图片')
    try: raw=base64.b64decode(encoded,validate=True)
    except (ValueError,binascii.Error): raise ValueError('图片编码无效')
    if not raw or len(raw)>5*1024*1024: raise ValueError('图片文件需小于 5 MB')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ['PNG','JPEG','WEBP']: raise ValueError('支持 PNG、JPEG 或 WebP')
            if image.width*image.height>20_000_000: raise ValueError('图片需小于 2000 万像素')
            image.load(); result=image.convert('RGB')
    except (OSError,Image.DecompressionBombError): raise ValueError('无法读取图片')
    edge=payload.get('max_image_edge',512)
    if type(edge) is not int or edge not in [512,768]: raise ValueError('分辨率需选择 512 或 768')
    return result,question.strip(),edge
